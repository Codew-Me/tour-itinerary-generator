"""Database query repositories."""

from __future__ import annotations

from sqlalchemy import func, or_
from sqlalchemy.orm import Session, joinedload

from src.data.normalizer import normalize_district, normalize_text
from src.database.models import Attraction, Destination, District


class AttractionRepository:
    def __init__(self, session: Session):
        self.session = session

    def search(
        self,
        query: str | None = None,
        district: str | None = None,
        destination: str | None = None,
        category: str | None = None,
        mood: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        q = (
            self.session.query(Attraction)
            .join(Destination)
            .join(District)
            .options(joinedload(Attraction.destination).joinedload(Destination.district))
        )

        if district:
            dnorm = normalize_text(normalize_district(district))
            q = q.filter(
                or_(
                    func.lower(District.normalized_name) == dnorm,
                    func.lower(District.name) == dnorm,
                )
            )

        if destination:
            dest_norm = normalize_text(destination)
            q = q.filter(
                or_(
                    func.lower(Destination.normalized_name).contains(dest_norm),
                    func.lower(Destination.name).contains(dest_norm),
                    func.lower(Attraction.normalized_name).contains(dest_norm),
                    func.lower(Attraction.name).contains(dest_norm),
                )
            )

        if category:
            q = q.filter(func.lower(Attraction.category) == category.lower().strip())

        if mood:
            q = q.filter(func.lower(Attraction.mood) == mood.lower().strip())

        if query:
            qnorm = normalize_text(query)
            q = q.filter(
                or_(
                    Attraction.normalized_name.contains(qnorm),
                    func.lower(Attraction.category).contains(qnorm),
                    func.lower(Attraction.mood).contains(qnorm),
                    func.lower(Attraction.details).contains(qnorm),
                    func.lower(Destination.name).contains(qnorm),
                )
            )

        return [self._to_dict(a) for a in q.limit(limit).all()]

    def get_by_id(self, attraction_id: int) -> dict | None:
        attraction = (
            self.session.query(Attraction)
            .join(Destination)
            .join(District)
            .options(joinedload(Attraction.destination).joinedload(Destination.district))
            .filter(Attraction.id == attraction_id)
            .first()
        )
        return self._to_dict(attraction) if attraction else None

    def get_by_name(self, name: str) -> dict | None:
        norm = normalize_text(name)
        attraction = (
            self.session.query(Attraction)
            .join(Destination)
            .join(District)
            .options(joinedload(Attraction.destination).joinedload(Destination.district))
            .filter(
                or_(
                    func.lower(Attraction.name) == name.lower(),
                    Attraction.normalized_name == norm,
                )
            )
            .first()
        )
        return self._to_dict(attraction) if attraction else None

    def list_by_district(self, district: str) -> list[dict]:
        return self.search(district=district, limit=200)

    def list_category_stats(self) -> list[dict]:
        """Category counts from the attractions dataset (source of truth for trip themes)."""
        from src.services.preferences import VALID_CATEGORIES

        rows = (
            self.session.query(Attraction.category, func.count(Attraction.id))
            .group_by(Attraction.category)
            .all()
        )
        counts = {cat: int(n) for cat, n in rows if cat}
        total = sum(counts.values())
        ordered: list[dict] = []
        seen: set[str] = set()
        for cat in VALID_CATEGORIES:
            count = counts.get(cat, 0)
            ordered.append({"name": cat, "count": count})
            seen.add(cat)
        for cat, count in sorted(counts.items()):
            if cat not in seen:
                ordered.append({"name": cat, "count": count})
        return ordered, total

    def get_destination_summary(self, destination_name: str) -> dict | None:
        norm = normalize_text(destination_name)
        dest = (
            self.session.query(Destination)
            .join(District)
            .options(joinedload(Destination.district))
            .filter(
                or_(
                    func.lower(Destination.name) == destination_name.lower(),
                    Destination.normalized_name == norm,
                )
            )
            .first()
        )
        if not dest:
            attraction = self.get_by_name(destination_name)
            if attraction:
                dest = self.session.get(Destination, attraction["destination_id"])
            else:
                return None

        attractions = self.session.query(Attraction).filter(Attraction.destination_id == dest.id).all()
        review_count = sum(a.review_count for a in attractions)

        return {
            "destination": dest.name,
            "district": dest.district.name,
            "destination_id": dest.id,
            "attraction_count": len(attractions),
            "review_count": review_count,
            "review_evidence_available": review_count > 0,
            "categories": sorted({a.category for a in attractions}),
            "moods": sorted({a.mood for a in attractions}),
            "attractions": [
                {
                    "name": a.name,
                    "category": a.category,
                    "mood": a.mood,
                    "review_count": a.review_count,
                    "review_evidence_available": a.review_evidence_available,
                }
                for a in attractions
            ],
        }

    @staticmethod
    def _to_dict(attraction: Attraction | None) -> dict:
        if not attraction:
            return {}
        dest = attraction.destination
        district = dest.district if dest else None
        return {
            "id": attraction.id,
            "name": attraction.name,
            "category": attraction.category,
            "mood": attraction.mood,
            "details": attraction.details,
            "image_url": attraction.image_url,
            "destination": dest.name if dest else None,
            "destination_id": dest.id if dest else None,
            "district": district.name if district else None,
            "review_count": attraction.review_count,
            "review_evidence_available": attraction.review_evidence_available,
            "match_type": attraction.match_type,
        }


class DistrictRepository:
    def __init__(self, session: Session):
        self.session = session

    def list_all(self) -> list[dict]:
        districts = self.session.query(District).order_by(District.name).all()
        return [{"id": d.id, "name": d.name} for d in districts]
