"""Attraction-first recommendation with reviews as supporting evidence only."""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.data.normalizer import normalize_text
from src.database.repositories import AttractionRepository
from src.services.category_filter import filter_by_category, filter_by_mood
from src.services.preferences import TourismPreferences
from src.services.review_evidence import get_evidence_for_attraction
from src.services.review_stats import enrich_attraction, get_destination_review_stats

INTEREST_DETAIL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "adventure": ("adventure", "explore", "climbing", "climb", "cave", "caves", "trek", "hike", "steep", "historic"),
    "history": ("historic", "history", "heritage", "ancient", "temple", "ruins", "archaeolog"),
    "nature": ("nature", "wildlife", "forest", "bird", "scenic", "views"),
    "relaxation": ("peaceful", "quiet", "calm", "relax", "serene"),
    "culture": ("cultural", "culture", "tradition", "local"),
}


class RecommendationService:
    def __init__(self, session: Session):
        self.session = session
        self.repo = AttractionRepository(session)

    def recommend(
        self,
        user_message: str,
        preferences: TourismPreferences | None = None,
        limit: int = 8,
        exclude_ids: list[int] | None = None,
        dislikes: list[str] | None = None,
        pace: str | None = None,
        duration_days: int | None = None,
        boost_review_popularity: bool = False,
        category_strict: bool | None = None,
        mood_strict: bool | None = None,
    ) -> dict:
        prefs = preferences or TourismPreferences()
        strict_category = (
            category_strict
            if category_strict is not None
            else bool(prefs.category_confirmed and prefs.category)
        )
        strict_mood = (
            mood_strict
            if mood_strict is not None
            else bool(prefs.mood_confirmed and prefs.mood)
        )
        evidence_query = self._evidence_query(prefs)
        exclude_ids = set(exclude_ids or [])
        dislikes = [d.lower() for d in (dislikes or [])]

        # STEP 1: Search attractions ONLY (primary source)
        search_limit = min(max(limit * 4, 50), 350)
        attractions = self.repo.search(
            query=prefs.search_query,
            category=prefs.category,
            mood=prefs.mood,
            district=prefs.district,
            destination=prefs.destination,
            limit=search_limit,
        )

        # Broaden only when filters are not confirmed hard constraints
        if not strict_category and not strict_mood:
            if len(attractions) < 3 and prefs.desired_experience and not prefs.search_query:
                attractions = self.repo.search(
                    mood=prefs.mood,
                    district=prefs.district,
                    limit=50,
                )

            if len(attractions) < 3 and prefs.mood:
                attractions = self.repo.search(mood=prefs.mood, limit=50)

            if len(attractions) < 3 and prefs.category:
                attractions = self.repo.search(
                    category=prefs.category,
                    district=prefs.district,
                    limit=50,
                )
        elif strict_category and prefs.category and len(attractions) < 3:
            # Stay within category — widen geography, not category
            attractions = self.repo.search(category=prefs.category, limit=50)
        elif strict_mood and prefs.mood and len(attractions) < 3:
            # Stay within mood_tag — widen geography, not mood
            attractions = self.repo.search(mood=prefs.mood, limit=50)

        # Hard filters when user confirmed category tab or stated mood_tag
        attractions = filter_by_category(attractions, prefs.category, strict=strict_category)
        attractions = filter_by_mood(attractions, prefs.mood, strict=strict_mood)

        # Filter dislikes and excluded IDs
        filtered = []
        for item in attractions:
            if item.get("id") in exclude_ids:
                continue
            if self._matches_dislikes(item, dislikes):
                continue
            filtered.append(item)
        attractions = filtered if filtered else [a for a in attractions if a.get("id") not in exclude_ids]

        # STEP 2: Rank attractions using structured data (no global review search)
        scored = []
        for item in attractions:
            scores = self._score_attraction(
                item,
                prefs,
                user_message,
                pace=pace,
                duration_days=duration_days,
                boost_review_popularity=boost_review_popularity,
                strict_category=strict_category,
                strict_mood=strict_mood,
            )
            if item.get("id") in exclude_ids:
                scores["total"] *= 0.05
            item = dict(item)
            item = enrich_attraction(item)
            item["scores"] = scores
            item["total_score"] = round(scores["total"], 3)
            scored.append(item)

        scored.sort(key=lambda x: x["total_score"], reverse=True)
        top = scored[:limit]

        # STEP 3: Attach review evidence ONLY for ranked candidates
        for cand in top:
            attraction_id = cand.get("id")
            if attraction_id and cand.get("review_evidence_available"):
                reviews = get_evidence_for_attraction(
                    self.session,
                    attraction_id,
                    preference_query=evidence_query,
                    k=2,
                )
                cand["sample_reviews"] = reviews
                if reviews:
                    cand["review_relevance"] = max(r["relevance_score"] for r in reviews)
                    scores = cand["scores"]
                    scores["review_evidence"] = round(min(cand["review_relevance"], 1.0), 2)
                    scores["total"] = self._final_score(scores)
                    cand["total_score"] = round(scores["total"], 3)
                else:
                    cand["sample_reviews"] = []
                    cand["review_relevance"] = 0.0
            else:
                cand["sample_reviews"] = []
                cand["review_relevance"] = 0.0

            cand["evidence_level"] = (
                "Review-supported" if cand.get("sample_reviews") else "Metadata-supported"
            )

        top.sort(key=lambda x: x["total_score"], reverse=True)
        return {
            "preferences": user_message,
            "inferred_category": prefs.category,
            "inferred_mood": prefs.mood,
            "inferred_district": prefs.district,
            "desired_experience": prefs.desired_experience,
            "candidates": top,
        }

    def diverse_sample(self, limit: int = 6) -> dict:
        """General tourism request — diverse attractions across categories."""
        seen_cats: set[str] = set()
        sample: list[dict] = []
        all_attractions = self.repo.search(limit=200)

        for item in all_attractions:
            cat = item.get("category")
            if cat and cat not in seen_cats:
                seen_cats.add(cat)
                sample.append(item)
            if len(sample) >= limit:
                break

        if len(sample) < limit:
            for item in all_attractions:
                if item not in sample:
                    sample.append(item)
                if len(sample) >= limit:
                    break

        for item in sample:
            item["evidence_level"] = (
                "Review-supported" if item.get("review_evidence_available") else "Metadata-supported"
            )
            if item.get("id") and item.get("review_evidence_available"):
                item["sample_reviews"] = get_evidence_for_attraction(
                    self.session, item["id"], preference_query="travel experience", k=1
                )
            else:
                item["sample_reviews"] = []

        return {"candidates": sample, "diverse": True}

    @staticmethod
    def _evidence_query(prefs: TourismPreferences) -> str:
        terms = []
        if prefs.experience_description:
            terms.append(prefs.experience_description)
        terms.extend(prefs.desired_experience)
        if prefs.mood:
            terms.append(prefs.mood.lower())
        if prefs.category:
            terms.append(prefs.category.lower())
        return " ".join(terms) if terms else "travel experience scenic nature culture"

    def _score_attraction(
        self,
        item: dict,
        prefs: TourismPreferences,
        user_message: str,
        *,
        pace: str | None = None,
        duration_days: int | None = None,
        boost_review_popularity: bool = False,
        strict_category: bool = False,
        strict_mood: bool = False,
    ) -> dict:
        msg = user_message.lower()
        if strict_mood and prefs.mood:
            mood_match = 1.0 if item.get("mood") == prefs.mood else 0.0
        else:
            mood_match = 1.0 if prefs.mood and item.get("mood") == prefs.mood else 0.2
        if strict_category and prefs.category:
            category_match = 1.0 if item.get("category") == prefs.category else 0.0
        else:
            category_match = 1.0 if prefs.category and item.get("category") == prefs.category else 0.2
        interest_score = self._interest_score(item, prefs.desired_experience)
        location_match = 0.3
        if prefs.district and item.get("district"):
            if normalize_text(prefs.district) == normalize_text(item["district"]):
                location_match = 1.0
            elif normalize_text(prefs.district) in normalize_text(item.get("destination") or ""):
                location_match = 0.8

        semantic_match = 0.3
        details = (item.get("details") or "").lower()
        name = (item.get("name") or "").lower()
        category = (item.get("category") or "").lower()
        mood = (item.get("mood") or "").lower()

        # Semantic match from LLM experience description
        exp_text = (prefs.experience_description or prefs.search_query or "").lower()
        if exp_text:
            exp_words = [w for w in exp_text.split() if len(w) > 3]
            hits = sum(1 for w in exp_words if w in details or w in name or w in category or w in mood)
            if hits:
                semantic_match = min(1.0, 0.3 + hits * 0.15)

        for term in prefs.desired_experience:
            if term.lower() in details or term.lower() in name:
                semantic_match = min(1.0, semantic_match + 0.15)

        data_quality = 0.5
        if item.get("category") and item.get("mood"):
            data_quality = 0.8
        if item.get("details") and len(item.get("details", "")) > 100:
            data_quality = 1.0

        review_bonus = 0.3 if item.get("review_evidence_available") else 0.0

        # Review dataset stats (precomputed)
        dest_stats = get_destination_review_stats(item.get("destination") or item.get("name"))
        review_signal = 0.0
        themes: dict = {}
        if dest_stats:
            count = dest_stats.get("review_count", 0)
            review_signal = min(1.0, 0.3 + (count ** 0.5) / 20)
            themes = dest_stats.get("themes", {})
        if prefs.category == "Heritage" and themes.get("historic", 0) > 2:
            review_signal = min(1.0, review_signal + 0.15)
        if prefs.category == "Wild" and themes.get("wildlife", 0) > 2:
            review_signal = min(1.0, review_signal + 0.15)
        for interest in prefs.desired_experience:
            il = interest.lower()
            if il in ("adventure", "adventurous") and (
                themes.get("adventure", 0) > 1
                or themes.get("climb", 0) > 1
                or themes.get("exploration", 0) > 1
            ):
                review_signal = min(1.0, review_signal + 0.12)
            if il in ("relaxation", "relaxed", "peaceful") and themes.get("peaceful", 0) > 1:
                review_signal = min(1.0, review_signal + 0.12)

        duration_fit = 0.5
        if duration_days is not None:
            if duration_days <= 2 and location_match >= 0.8:
                duration_fit = 1.0
            elif duration_days >= 5 and review_signal > 0.5:
                duration_fit = 0.9

        pace_fit = 0.5
        if pace == "relaxed":
            themes = (dest_stats or {}).get("themes", {})
            if themes.get("crowded", 0) > themes.get("peaceful", 0):
                pace_fit = 0.2
            else:
                pace_fit = 0.8
        elif pace == "packed":
            pace_fit = 0.7

        popularity_weight = 0.12 if boost_review_popularity else 0.06

        if strict_category or strict_mood:
            total = (
                category_match * (0.40 if strict_category else 0.18)
                + mood_match * (0.40 if strict_mood else 0.12)
                + interest_score * 0.14
                + location_match * 0.14
                + semantic_match * 0.06
                + data_quality * 0.04
                + review_bonus * 0.04
                + review_signal * popularity_weight
                + duration_fit * 0.06
                + pace_fit * 0.08
            )
        else:
            total = (
                mood_match * 0.22
                + category_match * 0.22
                + location_match * 0.18
                + semantic_match * 0.12
                + data_quality * 0.04
                + review_bonus * 0.04
                + review_signal * popularity_weight
                + duration_fit * 0.08
                + pace_fit * 0.08
            )
        return {
            "mood_match": round(mood_match, 2),
            "category_match": round(category_match, 2),
            "interest_score": round(interest_score, 2),
            "location_match": round(location_match, 2),
            "semantic_match": round(semantic_match, 2),
            "data_quality": round(data_quality, 2),
            "review_evidence": 0.0,
            "review_signal": round(review_signal, 2),
            "total": total,
        }

    @staticmethod
    def _interest_score(item: dict, interests: list[str]) -> float:
        if not interests:
            return 0.5
        details = (item.get("details") or "").lower()
        name = (item.get("name") or "").lower()
        blob = f"{name} {details}"
        dest_stats = get_destination_review_stats(item.get("destination") or item.get("name"))
        themes = (dest_stats or {}).get("themes", {})
        score = 0.35
        hits = 0
        for interest in interests:
            il = interest.lower()
            keywords = INTEREST_DETAIL_KEYWORDS.get(il, (il,))
            if any(kw in blob for kw in keywords):
                hits += 1
            if il in ("adventure", "adventurous") and (
                themes.get("adventure", 0) > 0
                or themes.get("climb", 0) > 0
                or themes.get("exploration", 0) > 0
            ):
                hits += 1
            if il in ("relaxation", "relaxed", "peaceful") and themes.get("peaceful", 0) > 0:
                hits += 1
        if hits:
            score = min(1.0, 0.45 + hits * 0.18)
        return score

    @staticmethod
    def _matches_dislikes(item: dict, dislikes: list[str]) -> bool:
        if not dislikes:
            return False
        blob = " ".join([
            str(item.get("name") or ""),
            str(item.get("details") or ""),
            str(item.get("category") or ""),
        ]).lower()
        for d in dislikes:
            if d in blob:
                return True
        return False

    @staticmethod
    def _final_score(scores: dict) -> float:
        return (
            scores.get("mood_match", 0) * 0.28
            + scores.get("category_match", 0) * 0.23
            + scores.get("location_match", 0) * 0.18
            + scores.get("semantic_match", 0) * 0.14
            + scores.get("data_quality", 0) * 0.05
            + scores.get("review_evidence", 0) * 0.12
        )
