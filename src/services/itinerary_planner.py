"""Structured itinerary planning: retrieve → validate → geo-filter → route → schedule."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from src.data.normalizer import normalize_text
from src.services.category_filter import filter_by_category, filter_by_mood, is_category_compatible
from src.services.conversation_state import ConversationState
from src.services.geography import (
    DEFAULT_MAX_RADIUS_KM,
    distance_between_districts,
    estimate_travel_minutes,
    get_district_coords,
    haversine_km,
    is_geographically_reasonable,
    normalize_district_key,
    resolve_routing_anchor,
)
from src.services.preferences import TourismPreferences
from src.services.recommendation_service import RecommendationService
from src.services.review_evidence import get_evidence_for_attraction
from src.services.review_stats import enrich_attraction, resolve_district_from_reviews
from src.database.repositories import AttractionRepository

# Visit duration estimates by category (minutes) — clearly estimates, not from dataset.
VISIT_DURATION_BY_CATEGORY: dict[str, int] = {
    "Wild": 180,
    "Heritage": 120,
    "Scenic": 120,
    "Pristine": 150,
    "Thrills": 120,
    "Essence": 90,
}
DEFAULT_VISIT_MINUTES = 120
DAILY_ACTIVITY_BUDGET_MINUTES = 540  # ~9 hours incl. breaks
DAILY_BREAK_MINUTES = 60
MAX_STOPS_PER_DAY = 3
CANDIDATE_POOL_SIZE = 30
MAX_PLAN_ATTRACTIONS = 15
DATASET_ATTRACTION_CAP = 350


def max_stops_for_pace(pace: str | None) -> int:
    if pace == "relaxed":
        return 2
    if pace == "packed":
        return MAX_STOPS_PER_DAY
    return 2


def plan_attraction_cap(num_days: int, pace: str | None) -> int:
    """How many distinct stops a trip of this length can use."""
    return min(num_days * max_stops_for_pace(pace), DATASET_ATTRACTION_CAP)


def candidate_pool_size(num_days: int, pace: str | None) -> int:
    """Retrieve more candidates for longer trips."""
    cap = plan_attraction_cap(num_days, pace)
    return min(max(cap * 2, CANDIDATE_POOL_SIZE), DATASET_ATTRACTION_CAP)


def geo_filter_radius_km(num_days: int) -> float:
    """Short trips stay local; longer trips can cover the island."""
    if num_days <= 3:
        return DEFAULT_MAX_RADIUS_KM
    if num_days <= 7:
        return 120.0
    if num_days <= 14:
        return 200.0
    return 350.0


@dataclass
class ItineraryStop:
    attraction_id: int
    name: str
    district: str
    destination: str | None
    category: str
    mood: str
    slot: str  # Morning | Afternoon | Evening
    visit_duration_minutes: int
    travel_from_previous_minutes: int | None
    travel_distance_km: float | None
    why: str
    evidence_level: str
    review_snippet: str | None = None
    details_excerpt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ItineraryDay:
    day: int
    stops: list[ItineraryStop] = field(default_factory=list)
    estimated_total_minutes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "estimated_total_minutes": self.estimated_total_minutes,
            "stops": [s.to_dict() for s in self.stops],
        }


@dataclass
class ItineraryPlan:
    duration_days: int
    starting_location: str | None
    mood: str | None
    experience: str | None
    days: list[ItineraryDay] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    validation_passed: bool = True
    validation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_days": self.duration_days,
            "starting_location": self.starting_location,
            "mood": self.mood,
            "experience": self.experience,
            "days": [d.to_dict() for d in self.days],
            "assumptions": self.assumptions,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
        }


class ItineraryPlanner:
    def __init__(self, session: Session):
        self.session = session
        self.recommender = RecommendationService(session)
        self.repo = AttractionRepository(session)

    def build_plan(
        self,
        state: ConversationState,
        user_message: str = "",
    ) -> ItineraryPlan:
        if state.itinerary_modify and state.current_itinerary and state.itinerary_modify_mode:
            return self._modify_existing_plan(state, user_message)

        days = state.duration_days or 3
        starting = state.starting_location or state.district
        prefs = state.to_tourism_preferences()
        if starting and not prefs.district:
            prefs.district = starting

        plan = ItineraryPlan(
            duration_days=days,
            starting_location=starting,
            mood=state.mood,
            experience=state.experience,
        )

        # Use previously recommended attractions when user asks to "choose some and plan"
        if state.use_previous_recommendations and state.last_recommendations:
            candidates = self._hydrate_candidates(state.last_recommendations)
            if state.selected_attraction_ids:
                selected = set(state.selected_attraction_ids)
                candidates = [c for c in candidates if c["id"] in selected]
            candidates = self._validate_candidates(candidates)
            candidates = self._apply_category_gate(candidates, state)
            candidates = self._deduplicate(candidates)

            if not starting and candidates:
                starting = self._infer_starting_location(candidates)
                plan.starting_location = starting

            if candidates:
                plan.assumptions.append(
                    "Built from the places we discussed in your previous recommendations."
                )
                self._attach_evidence(candidates, prefs)
                plan.days, trim_note = self._schedule_days_from_suggestions(
                    candidates, days, starting, state, district=state.district
                )
                if trim_note:
                    plan.assumptions.append(trim_note)
                errors = self._validate_plan(plan)
                plan.validation_passed = len(errors) == 0
                plan.validation_errors = errors
                if starting:
                    plan.assumptions.append(
                        "Distances and travel times are estimates based on district centroids (~45 km/h average)."
                    )
                return plan

        if not starting:
            plan.assumptions.append(
                "No starting location specified — attractions are grouped by nearest sensible district clusters."
            )

        # 1. Retrieve large candidate pool
        candidates = self._retrieve_candidates(user_message, prefs, starting, days, state)

        # Must-include attractions by name
        for term in state.must_include:
            found = self.repo.search(query=term, limit=3)
            for f in found:
                if normalize_text(term) in normalize_text(f.get("name") or ""):
                    if f["id"] not in {c["id"] for c in candidates}:
                        candidates.insert(0, f)
                    break

        # 2. Validate candidates exist in dataset + category gate
        candidates = self._validate_candidates(candidates)
        candidates = self._apply_category_gate(candidates, state)
        candidates = self._deduplicate(candidates)

        attraction_cap = plan_attraction_cap(days, state.pace)

        # 3. Geographic filtering (radius scales with trip length)
        pre_geo = candidates
        if starting:
            for radius in (geo_filter_radius_km(days), 250.0, 350.0):
                candidates = self._geo_filter(pre_geo, starting, max_radius_km=radius)
                if len(candidates) >= attraction_cap or radius >= 350.0:
                    break
            if not candidates:
                plan.validation_passed = False
                plan.validation_errors.append(
                    f"No attractions found within reasonable distance of {starting}."
                )
                return plan

        # 4. Rank and cap
        candidates = candidates[:attraction_cap]

        # 5. Attach review evidence (supporting only)
        self._attach_evidence(candidates, prefs)

        # 6. Route optimize + daily schedule
        plan.days, trim_note = self._schedule_days_from_suggestions(
            candidates, days, starting, state, district=state.district
        )
        if trim_note:
            plan.assumptions.append(trim_note)

        # 7. Final category validation + replacement
        plan.days = self._enforce_category_on_plan(plan.days, state, starting, days)

        if len(plan.days) < days:
            plan.assumptions.append(
                f"Filled {len(plan.days)} of {days} requested days from matching places in our dataset."
            )

        # 8. Validate final plan
        errors = self._validate_plan(plan)
        if errors:
            plan.validation_passed = False
            plan.validation_errors = errors
            # Retry once with stricter geo filter if duplicates or geo issues
            if any("duplicate" in e.lower() or "distance" in e.lower() for e in errors):
                retry_radius = (
                    geo_filter_radius_km(days) * 0.85
                    if days >= 4
                    else DEFAULT_MAX_RADIUS_KM * 0.7
                )
                tighter = self._geo_filter(
                    self._deduplicate(self._validate_candidates(
                        self._retrieve_candidates(user_message, prefs, starting, days, state)
                    )),
                    starting,
                    max_radius_km=retry_radius,
                )[:attraction_cap]
                if tighter:
                    tighter = self._apply_category_gate(tighter, state)
                    self._attach_evidence(tighter, prefs)
                    plan.days, trim_note = self._schedule_days_from_suggestions(
                        tighter, days, starting, state, district=state.district
                    )
                    if trim_note:
                        plan.assumptions.append(trim_note)
                    plan.days = self._enforce_category_on_plan(plan.days, state, starting, days)
                    errors = self._validate_plan(plan)
                    plan.validation_passed = len(errors) == 0
                    plan.validation_errors = errors

        if starting:
            plan.assumptions.append(
                f"Distances and travel times are estimates based on district centroids (~{45} km/h average)."
            )

        return plan

    def _retrieve_candidates(
        self,
        user_message: str,
        prefs: TourismPreferences,
        starting: str | None,
        days: int,
        state: ConversationState,
    ) -> list[dict]:
        limit = candidate_pool_size(days, state.pace)
        needed = plan_attraction_cap(days, state.pace)
        exclude = list(
            dict.fromkeys(
                (state.already_recommended_ids if state.itinerary_modify else [])
                + state.rejected_attraction_ids
            )
        )
        # Primary search with preferences
        result = self.recommender.recommend(
            user_message,
            preferences=prefs,
            limit=limit,
            exclude_ids=exclude,
            dislikes=state.dislikes,
            pace=state.pace,
            duration_days=days,
            category_strict=bool(state.category_confirmed and prefs.category),
            mood_strict=bool(state.mood_confirmed and prefs.mood),
        )

        candidates = list(result.get("candidates", []))

        # Broaden pool within confirmed category or mood only
        if len(candidates) < days * 2 and prefs.category and state.category_confirmed:
            extra = self.recommender.recommend(
                user_message,
                preferences=TourismPreferences(category=prefs.category, category_confirmed=True),
                limit=limit,
                exclude_ids=exclude,
                dislikes=state.dislikes,
                pace=state.pace,
                duration_days=days,
                category_strict=True,
            )
            seen = {c["id"] for c in candidates}
            for c in extra.get("candidates", []):
                if c["id"] not in seen:
                    candidates.append(c)
                    seen.add(c["id"])
        if len(candidates) < days * 2 and prefs.mood and state.mood_confirmed:
            extra = self.recommender.recommend(
                user_message,
                preferences=TourismPreferences(mood=prefs.mood, mood_confirmed=True),
                limit=limit,
                exclude_ids=exclude,
                dislikes=state.dislikes,
                pace=state.pace,
                duration_days=days,
                mood_strict=True,
            )
            seen = {c["id"] for c in candidates}
            for c in extra.get("candidates", []):
                if c["id"] not in seen:
                    candidates.append(c)
                    seen.add(c["id"])

        if len(candidates) < needed:
            seen = {c["id"] for c in candidates}
            broad = self.repo.search(limit=min(needed * 3, DATASET_ATTRACTION_CAP))
            for item in broad:
                if item.get("id") in seen or item.get("id") in exclude:
                    continue
                seen.add(item["id"])
                candidates.append(item)
                if len(candidates) >= needed:
                    break

        return self._apply_dataset_gates(candidates, state)

    def _hydrate_candidates(self, stored: list[dict]) -> list[dict]:
        """Reload full attraction records from DB by ID."""
        hydrated: list[dict] = []
        for item in stored:
            aid = item.get("id")
            if not aid:
                continue
            full = self.repo.get_by_id(aid)
            if full:
                if item.get("total_score") is not None:
                    full["total_score"] = item["total_score"]
                hydrated.append(full)
            elif item.get("name") and item.get("district"):
                hydrated.append(dict(item))
        return hydrated

    @staticmethod
    def _infer_starting_location(candidates: list[dict]) -> str | None:
        """Pick the most common district among recommended candidates as route anchor."""
        counts: dict[str, int] = {}
        for c in candidates:
            d = c.get("district")
            if d:
                counts[d] = counts.get(d, 0) + 1
        if not counts:
            return None
        return max(counts, key=counts.get)

    @staticmethod
    def _validate_candidates(candidates: list[dict]) -> list[dict]:
        valid = []
        for c in candidates:
            if not c.get("id") or not c.get("name") or not c.get("district"):
                continue
            valid.append(c)
        return valid

    @staticmethod
    def _geo_filter(
        candidates: list[dict],
        starting: str,
        max_radius_km: float = DEFAULT_MAX_RADIUS_KM,
    ) -> list[dict]:
        anchor, _ = resolve_routing_anchor(starting, None)
        origin = anchor or starting
        filtered = [
            c for c in candidates
            if is_geographically_reasonable(origin, c.get("district"), max_radius_km)
        ]
        # Sort by distance from start
        def sort_key(c: dict) -> float:
            d = distance_between_districts(origin, c.get("district"))
            return d if d is not None else 9999.0

        filtered.sort(key=sort_key)
        return filtered

    @staticmethod
    def _deduplicate(candidates: list[dict]) -> list[dict]:
        seen_ids: set[int] = set()
        seen_names: set[str] = set()
        out = []
        for c in candidates:
            aid = c["id"]
            name_key = normalize_text(c["name"])
            if aid in seen_ids or name_key in seen_names:
                continue
            seen_ids.add(aid)
            seen_names.add(name_key)
            out.append(c)
        return out

    def _attach_evidence(self, candidates: list[dict], prefs: TourismPreferences) -> None:
        query = prefs.experience_description or "travel experience"
        for c in candidates:
            if c.get("review_evidence_available"):
                reviews = get_evidence_for_attraction(
                    self.session, c["id"], preference_query=query, k=1
                )
                c["sample_reviews"] = reviews
                c["evidence_level"] = "Review-supported" if reviews else "Metadata-supported"
            else:
                c["sample_reviews"] = []
                c["evidence_level"] = "Metadata-supported"

    @staticmethod
    def _apply_dataset_gates(candidates: list[dict], state: ConversationState) -> list[dict]:
        if state.category_tag and state.category_confirmed:
            candidates = filter_by_category(candidates, state.category_tag, strict=True)
        if state.mood_tag and state.mood_confirmed:
            candidates = filter_by_mood(candidates, state.mood_tag, strict=True)
        return candidates

    @staticmethod
    def _apply_category_gate(candidates: list[dict], state: ConversationState) -> list[dict]:
        return ItineraryPlanner._apply_dataset_gates(candidates, state)

    def _modify_existing_plan(
        self,
        state: ConversationState,
        user_message: str,
    ) -> ItineraryPlan:
        """Adjust the stored itinerary instead of regenerating from scratch."""
        existing = state.current_itinerary or {}
        plan = ItineraryPlan(
            duration_days=existing.get("duration_days") or state.duration_days or 3,
            starting_location=state.starting_location or existing.get("starting_location"),
            mood=state.mood,
            experience=state.experience or state.category_tag,
        )
        days_data = existing.get("days") or []
        used_ids = {
            stop["attraction_id"]
            for day in days_data
            for stop in day.get("stops", [])
            if stop.get("attraction_id")
        }

        mode = state.itinerary_modify_mode or "add"
        if mode == "relaxed":
            plan.days = self._relax_plan_days(days_data, state)
            plan.assumptions.append("Reduced stops to make the itinerary more relaxed.")
        elif mode == "focus":
            plan.days = self._focus_plan_days(days_data, state, user_message, used_ids)
            plan.assumptions.append(
                f"Re-ranked stops to emphasize your {', '.join(state.interests) or 'interests'}."
            )
        else:  # add
            plan.days = self._add_to_plan_days(days_data, state, user_message, used_ids)
            plan.assumptions.append("Kept your current itinerary and added more matching places.")

        plan.days = self._enforce_category_on_plan(
            plan.days,
            state,
            plan.starting_location,
            plan.duration_days,
        )
        errors = self._validate_plan(plan)
        plan.validation_passed = len(errors) == 0
        plan.validation_errors = errors
        if plan.starting_location:
            plan.assumptions.append(
                "Distances and travel times are estimates based on district centroids (~45 km/h average)."
            )
        return plan

    def _plan_days_from_dict(self, days_data: list[dict], state: ConversationState) -> list[ItineraryDay]:
        days: list[ItineraryDay] = []
        for d in days_data:
            day = ItineraryDay(day=d.get("day", len(days) + 1))
            for i, stop in enumerate(d.get("stops", [])):
                aid = stop.get("attraction_id")
                full = self.repo.get_by_id(aid) if aid else None
                if full:
                    primary = day.stops[0].district if day.stops else state.starting_location
                    day.stops.append(self._candidate_to_stop(full, state, i, primary))
                else:
                    day.stops.append(ItineraryStop(
                        attraction_id=stop.get("attraction_id", 0),
                        name=stop.get("name", "Unknown"),
                        district=stop.get("district", "—"),
                        destination=stop.get("destination"),
                        category=stop.get("category", "—"),
                        mood=stop.get("mood", "—"),
                        slot=stop.get("slot", "Morning"),
                        visit_duration_minutes=stop.get("visit_duration_minutes", 120),
                        travel_from_previous_minutes=stop.get("travel_from_previous_minutes"),
                        travel_distance_km=stop.get("travel_distance_km"),
                        why=stop.get("why", "Previously selected"),
                        evidence_level=stop.get("evidence_level", "Metadata-supported"),
                        review_snippet=stop.get("review_snippet"),
                        details_excerpt=stop.get("details_excerpt"),
                    ))
            day.estimated_total_minutes = d.get("estimated_total_minutes", 0)
            days.append(day)
        return days

    def _relax_plan_days(self, days_data: list[dict], state: ConversationState) -> list[ItineraryDay]:
        days = self._plan_days_from_dict(days_data, state)
        relaxed: list[ItineraryDay] = []
        for day in days:
            new_day = ItineraryDay(day=day.day)
            if day.stops:
                new_day.stops = day.stops[:1]
                new_day.estimated_total_minutes = sum(
                    s.visit_duration_minutes + (s.travel_from_previous_minutes or 0)
                    for s in new_day.stops
                )
            relaxed.append(new_day)
        return relaxed

    def _add_to_plan_days(
        self,
        days_data: list[dict],
        state: ConversationState,
        user_message: str,
        used_ids: set[int],
    ) -> list[ItineraryDay]:
        days = self._plan_days_from_dict(days_data, state)
        prefs = state.to_tourism_preferences()
        extra = self._retrieve_candidates(user_message, prefs, state.starting_location, len(days), state)
        extra = [c for c in extra if c.get("id") not in used_ids]
        extra = self._deduplicate(extra)
        if not extra:
            return days

        max_per_day = 2 if state.pace == "relaxed" else 3
        for day in days:
            if not extra:
                break
            if len(day.stops) >= max_per_day:
                continue
            primary = day.stops[0].district if day.stops else state.starting_location
            for cand in extra:
                cand_district = resolve_district_from_reviews(cand.get("destination"), cand.get("district"))
                if primary and not self._districts_compatible_for_day(primary, cand_district):
                    continue
                stop = self._candidate_to_stop(cand, state, len(day.stops), primary)
                day.stops.append(stop)
                used_ids.add(cand["id"])
                extra = [c for c in extra if c.get("id") != cand["id"]]
                if len(day.stops) >= max_per_day:
                    break
        return days

    def _focus_plan_days(
        self,
        days_data: list[dict],
        state: ConversationState,
        user_message: str,
        used_ids: set[int],
    ) -> list[ItineraryDay]:
        days = self._plan_days_from_dict(days_data, state)
        prefs = state.to_tourism_preferences()
        pool = self._retrieve_candidates(user_message, prefs, state.starting_location, len(days), state)
        pool = [c for c in pool if c.get("id") not in used_ids]
        pool.sort(key=lambda x: x.get("total_score", 0), reverse=True)

        for day in days:
            for i, stop in enumerate(day.stops):
                if not pool:
                    break
                current = self.repo.get_by_id(stop.attraction_id) or {}
                current_score = current.get("total_score", 0)
                best = pool[0]
                if best.get("total_score", 0) > current_score + 0.05:
                    primary = day.stops[0].district if day.stops else state.starting_location
                    cand_district = resolve_district_from_reviews(best.get("destination"), best.get("district"))
                    if primary and not self._districts_compatible_for_day(primary, cand_district):
                        pool.pop(0)
                        continue
                    day.stops[i] = self._candidate_to_stop(best, state, i, primary)
                    used_ids.add(best["id"])
                    pool.pop(0)
        return days

    def _candidate_to_stop(
        self,
        cand: dict,
        state: ConversationState,
        slot_index: int,
        prev_district: str | None,
    ) -> ItineraryStop:
        visit_mins = VISIT_DURATION_BY_CATEGORY.get(cand.get("category", ""), DEFAULT_VISIT_MINUTES)
        travel_km = None
        travel_mins = 0
        display_district = resolve_district_from_reviews(cand.get("destination"), cand.get("district"))
        if prev_district and slot_index > 0:
            travel_km = distance_between_districts(prev_district, display_district)
            if travel_km is not None:
                travel_mins = estimate_travel_minutes(travel_km)
        enriched = enrich_attraction(cand)
        snippet = enriched.get("review_summary")
        slot = ["Morning", "Afternoon", "Evening"][min(slot_index, 2)]
        return ItineraryStop(
            attraction_id=cand["id"],
            name=cand["name"],
            district=display_district,
            destination=cand.get("destination"),
            category=cand.get("category") or "—",
            mood=cand.get("mood") or "—",
            slot=slot,
            visit_duration_minutes=visit_mins,
            travel_from_previous_minutes=travel_mins if slot_index > 0 else None,
            travel_distance_km=round(travel_km, 1) if travel_km is not None and slot_index > 0 else None,
            why=self._why_selected(cand, state),
            evidence_level=cand.get("evidence_level", "Metadata-supported"),
            review_snippet=snippet,
            details_excerpt=(cand.get("details") or "")[:300] or None,
        )

    def _enforce_category_on_plan(
        self,
        days: list[ItineraryDay],
        state: ConversationState,
        starting: str | None,
        num_days: int,
    ) -> list[ItineraryDay]:
        if not state.category_tag or not state.category_confirmed:
            return days

        used_ids = {stop.attraction_id for day in days for stop in day.stops}
        replacements = self._retrieve_candidates(
            "",
            state.to_tourism_preferences(),
            starting,
            num_days,
            state,
        )
        replacements = [c for c in replacements if c.get("id") not in used_ids]

        cleaned: list[ItineraryDay] = []
        for day in days:
            new_day = ItineraryDay(day=day.day)
            primary = None
            for stop in day.stops:
                cand = self.repo.get_by_id(stop.attraction_id) or {"category": stop.category}
                if is_category_compatible(cand, state.category_tag):
                    new_day.stops.append(stop)
                    primary = primary or stop.district
                    continue
                replaced = False
                for alt in replacements:
                    alt_district = resolve_district_from_reviews(alt.get("destination"), alt.get("district"))
                    if primary and not self._districts_compatible_for_day(primary, alt_district):
                        continue
                    new_day.stops.append(
                        self._candidate_to_stop(alt, state, len(new_day.stops), primary)
                    )
                    used_ids.add(alt["id"])
                    replacements = [r for r in replacements if r.get("id") != alt["id"]]
                    primary = primary or alt_district
                    replaced = True
                    break
                if not replaced:
                    continue
            if new_day.stops:
                new_day.estimated_total_minutes = day.estimated_total_minutes
                cleaned.append(new_day)
        return cleaned

    @staticmethod
    def _districts_compatible_for_day(primary: str | None, other: str | None) -> bool:
        if not primary or not other:
            return True
        if normalize_district_key(primary) == normalize_district_key(other):
            return True
        return is_geographically_reasonable(primary, other, max_radius_km=45.0)

    def _schedule_days_from_suggestions(
        self,
        candidates: list[dict],
        num_days: int,
        starting: str | None,
        state: ConversationState,
        *,
        district: str | None = None,
    ) -> tuple[list[ItineraryDay], str | None]:
        """Route-order suggested places from the start, then pack into days."""
        if not candidates:
            return [], None

        max_stops = max_stops_for_pace(state.pace)
        capacity = num_days * max_stops
        trim_note: str | None = None

        ordered = self._route_optimize(candidates, starting, district)
        if len(ordered) > capacity:
            dropped = ordered[capacity:]
            ordered = ordered[:capacity]
            dropped_names = ", ".join(c["name"] for c in dropped[:3])
            if len(dropped) > 3:
                dropped_names += f", and {len(dropped) - 3} more"
            trim_note = (
                f"I can build these into the itinerary, but {len(candidates)} stops across these "
                f"destinations would make the trip quite rushed. I've kept the strongest {len(ordered)} "
                f"and left the others as optional stops ({dropped_names})."
            )

        anchor_district, _ = resolve_routing_anchor(starting, district)
        days: list[ItineraryDay] = []
        day_num = 1
        day = ItineraryDay(day=day_num)
        day_minutes = DAILY_BREAK_MINUTES
        prev_district = anchor_district

        for idx, cand in enumerate(ordered):
            display_district = resolve_district_from_reviews(cand.get("destination"), cand.get("district"))
            visit_mins = VISIT_DURATION_BY_CATEGORY.get(cand.get("category", ""), DEFAULT_VISIT_MINUTES)
            travel_km = None
            travel_mins = 0
            if prev_district:
                travel_km = distance_between_districts(prev_district, display_district)
                if travel_km is not None:
                    travel_mins = estimate_travel_minutes(travel_km)

            primary_district = day.stops[0].district if day.stops else None
            needs_new_day = (
                len(day.stops) >= max_stops
                or (
                    day.stops
                    and primary_district
                    and not self._districts_compatible_for_day(primary_district, display_district)
                )
                or (
                    day.stops
                    and day_minutes + travel_mins + visit_mins > DAILY_ACTIVITY_BUDGET_MINUTES
                )
            )
            if needs_new_day and day_num < num_days:
                if day.stops:
                    day.estimated_total_minutes = day_minutes
                    days.append(day)
                day_num += 1
                day = ItineraryDay(day=day_num)
                day_minutes = DAILY_BREAK_MINUTES
                prev_district = anchor_district if day_num == 1 else (days[-1].stops[-1].district if days else anchor_district)
                if prev_district:
                    travel_km = distance_between_districts(prev_district, display_district)
                    travel_mins = estimate_travel_minutes(travel_km) if travel_km is not None else 0

            primary_district = day.stops[0].district if day.stops else None
            if day.stops and primary_district and not self._districts_compatible_for_day(
                primary_district, display_district
            ):
                continue

            slot_index = len(day.stops)
            enriched = enrich_attraction(cand)
            stop = ItineraryStop(
                attraction_id=cand["id"],
                name=cand["name"],
                district=display_district,
                destination=cand.get("destination"),
                category=cand.get("category") or "—",
                mood=cand.get("mood") or "—",
                slot=["Morning", "Afternoon", "Evening"][min(slot_index, 2)],
                visit_duration_minutes=visit_mins,
                travel_from_previous_minutes=travel_mins if slot_index > 0 or day_num > 1 else None,
                travel_distance_km=round(travel_km, 1) if travel_km is not None and (slot_index > 0 or day_num > 1) else None,
                why=self._why_selected(cand, state),
                evidence_level=cand.get("evidence_level", "Metadata-supported"),
                review_snippet=enriched.get("review_summary"),
                details_excerpt=(cand.get("details") or "")[:300] or None,
            )
            day.stops.append(stop)
            day_minutes += travel_mins + visit_mins
            prev_district = display_district

        if day.stops:
            day.estimated_total_minutes = day_minutes
            days.append(day)

        return days, trim_note

    def _schedule_days(
        self,
        candidates: list[dict],
        num_days: int,
        starting: str | None,
        state: ConversationState,
    ) -> list[ItineraryDay]:
        if not candidates:
            return []

        max_stops = max_stops_for_pace(state.pace)
        target_total = min(len(candidates), num_days * max_stops)

        ranked = sorted(candidates, key=lambda x: x.get("total_score", 0), reverse=True)
        pool = self._route_optimize(ranked[: max(target_total * 3, num_days * 3)], starting, state.district)

        # Group by district cluster for coherent days
        by_district: dict[str, list[dict]] = {}
        district_labels: dict[str, str] = {}
        per_district_cap = max(max_stops * num_days, num_days)
        for cand in pool:
            label = resolve_district_from_reviews(cand.get("destination"), cand.get("district"))
            key = normalize_district_key(label)
            if key not in by_district:
                by_district[key] = []
                district_labels[key] = label
            if len(by_district[key]) < per_district_cap:
                by_district[key].append(cand)

        ordered_keys = self._order_district_clusters(
            list(by_district.keys()), starting, district_labels, routing_district=state.district
        )
        days: list[ItineraryDay] = []
        used_ids: set[int] = set()

        # Assign one primary district cluster per day; reuse districts if fewer clusters than days
        schedule_keys: list[str] = []
        if ordered_keys:
            for i in range(num_days):
                schedule_keys.append(ordered_keys[i % len(ordered_keys)])

        for day_num, dkey in enumerate(schedule_keys, start=1):
            day_candidates = [c for c in by_district.get(dkey, []) if c["id"] not in used_ids][:max_stops]
            if not day_candidates:
                # Pull next unused candidates from any district for remaining days
                for alt_key in ordered_keys:
                    extras = [c for c in by_district.get(alt_key, []) if c["id"] not in used_ids]
                    if extras:
                        day_candidates = extras[:max_stops]
                        dkey = alt_key
                        break
            if not day_candidates:
                continue

            day = ItineraryDay(day=day_num)
            day_minutes = DAILY_BREAK_MINUTES
            prev_district = starting if day_num == 1 else None

            for idx, cand in enumerate(day_candidates):
                if cand["id"] in used_ids:
                    continue
                visit_mins = VISIT_DURATION_BY_CATEGORY.get(
                    cand.get("category", ""), DEFAULT_VISIT_MINUTES
                )
                display_district = resolve_district_from_reviews(
                    cand.get("destination"), cand.get("district")
                )
                primary = day.stops[0].district if day.stops else None
                if day.stops and primary and not self._districts_compatible_for_day(
                    primary, display_district
                ):
                    continue
                travel_km = None
                travel_mins = 0
                if prev_district:
                    travel_km = distance_between_districts(prev_district, display_district)
                    if travel_km is not None:
                        travel_mins = estimate_travel_minutes(travel_km)

                if day_minutes + travel_mins + visit_mins > DAILY_ACTIVITY_BUDGET_MINUTES and idx > 0:
                    break

                enriched = enrich_attraction(cand)
                stop = ItineraryStop(
                    attraction_id=cand["id"],
                    name=cand["name"],
                    district=display_district,
                    destination=cand.get("destination"),
                    category=cand.get("category") or "—",
                    mood=cand.get("mood") or "—",
                    slot=["Morning", "Afternoon", "Evening"][idx],
                    visit_duration_minutes=visit_mins,
                    travel_from_previous_minutes=travel_mins if idx > 0 or day_num > 1 else None,
                    travel_distance_km=round(travel_km, 1) if travel_km is not None and (idx > 0 or day_num > 1) else None,
                    why=self._why_selected(cand, state),
                    evidence_level=cand.get("evidence_level", "Metadata-supported"),
                    review_snippet=enriched.get("review_summary"),
                    details_excerpt=(cand.get("details") or "")[:300] or None,
                )
                day.stops.append(stop)
                used_ids.add(cand["id"])
                day_minutes += travel_mins + visit_mins
                prev_district = display_district

            if day.stops:
                day.estimated_total_minutes = day_minutes
                days.append(day)

        return days

    @staticmethod
    def _order_district_clusters(
        district_keys: list[str],
        starting: str | None,
        labels: dict[str, str] | None = None,
        *,
        routing_district: str | None = None,
    ) -> list[str]:
        if not district_keys:
            return []
        anchor_district, start_coords = resolve_routing_anchor(starting, routing_district)
        if not anchor_district and not start_coords:
            return district_keys

        remaining = list(district_keys)
        ordered: list[str] = []
        current_district = anchor_district
        current_coords = start_coords

        while remaining:
            best_key = remaining[0]
            best_dist = float("inf")
            for key in remaining:
                label = (labels or {}).get(key, key)
                if current_district:
                    dist = distance_between_districts(current_district, label)
                elif current_coords:
                    coords = get_district_coords(label)
                    if coords:
                        dist = haversine_km(current_coords[0], current_coords[1], coords[0], coords[1])
                    else:
                        dist = None
                else:
                    dist = None
                score = dist if dist is not None else 9999.0
                if score < best_dist:
                    best_dist = score
                    best_key = key
            ordered.append(best_key)
            remaining.remove(best_key)
            current_district = (labels or {}).get(best_key, best_key)
            current_coords = get_district_coords(current_district)

        return ordered

    def _route_optimize(
        self,
        candidates: list[dict],
        starting: str | None,
        routing_district: str | None = None,
    ) -> list[dict]:
        """Nearest-neighbor ordering from the user's resolved start point."""
        if len(candidates) <= 1:
            return candidates

        anchor_district, start_coords = resolve_routing_anchor(starting, routing_district)
        remaining = list(candidates)
        ordered: list[dict] = []

        if start_coords:
            current_lat, current_lon = start_coords
        elif anchor_district:
            district_coords = get_district_coords(anchor_district)
            current_lat, current_lon = district_coords if district_coords else (6.927, 79.861)
        elif remaining:
            first_d = get_district_coords(remaining[0].get("district"))
            current_lat, current_lon = first_d if first_d else (6.927, 79.861)
        else:
            return []

        while remaining:
            best_idx = 0
            best_dist = float("inf")
            for i, c in enumerate(remaining):
                coords = get_district_coords(
                    resolve_district_from_reviews(c.get("destination"), c.get("district"))
                )
                if coords:
                    dist = haversine_km(current_lat, current_lon, coords[0], coords[1])
                else:
                    dist = 9999.0
                score = dist - (c.get("total_score", 0) * 5)
                if score < best_dist:
                    best_dist = score
                    best_idx = i
            chosen = remaining.pop(best_idx)
            ordered.append(chosen)
            coords = get_district_coords(
                resolve_district_from_reviews(chosen.get("destination"), chosen.get("district"))
            )
            if coords:
                current_lat, current_lon = coords

        return ordered

    @staticmethod
    def _why_selected(cand: dict, state: ConversationState) -> str:
        parts = []
        if state.category_tag and cand.get("category"):
            if state.category_tag.lower() == (cand.get("category") or "").lower():
                parts.append(f"Fits your {state.category_tag} focus")
        if state.travellers == "family":
            parts.append("Suitable pace for a family trip")
        if state.pace == "relaxed":
            parts.append("Works well on a relaxed day")
        if cand.get("district") and state.starting_location:
            if normalize_text(cand["district"]) == normalize_text(state.starting_location):
                parts.append("Near your starting area")
        if not parts:
            parts.append("Selected from ranked matches for your trip preferences")
        return ". ".join(parts) + "."

    @staticmethod
    def _validate_plan(plan: ItineraryPlan) -> list[str]:
        errors: list[str] = []
        seen_ids: set[int] = set()
        seen_names: set[str] = set()

        if not plan.days:
            errors.append("No days scheduled.")
            return errors

        for day in plan.days:
            if not day.stops:
                errors.append(f"Day {day.day} has no stops.")
            if len(day.stops) > MAX_STOPS_PER_DAY:
                errors.append(f"Day {day.day} has too many stops ({len(day.stops)}).")
            if day.estimated_total_minutes > DAILY_ACTIVITY_BUDGET_MINUTES + 60:
                errors.append(f"Day {day.day} schedule exceeds realistic daily budget.")

            for stop in day.stops:
                if stop.attraction_id in seen_ids:
                    errors.append(f"Duplicate attraction ID {stop.attraction_id} ({stop.name}).")
                seen_ids.add(stop.attraction_id)
                name_key = normalize_text(stop.name)
                if name_key in seen_names:
                    errors.append(f"Duplicate attraction name: {stop.name}.")
                seen_names.add(name_key)

                # Same-day districts must be geographically coherent
                if len(day.stops) > 1:
                    primary = day.stops[0].district
                    if not ItineraryPlanner._districts_compatible_for_day(primary, stop.district):
                        errors.append(
                            f"Day {day.day} mixes distant districts: {primary} and {stop.district}."
                        )

                if plan.starting_location and stop.district and plan.duration_days <= 3:
                    anchor, _ = resolve_routing_anchor(plan.starting_location, None)
                    origin = anchor or plan.starting_location
                    if not is_geographically_reasonable(
                        origin, stop.district, geo_filter_radius_km(plan.duration_days)
                    ):
                        errors.append(
                            f"{stop.name} ({stop.district}) is too far from {plan.starting_location}."
                        )

        return errors
