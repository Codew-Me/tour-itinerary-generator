"""Attraction card formatting."""

from src.services.attraction_cards import (
    format_attraction_card,
    format_details_for_card,
    format_recommendation_response,
    summarize_details,
)


class TestAttractionCards:
    def test_card_uses_location_not_district_label(self):
        card = format_attraction_card({
            "name": "Excel World Entertainment Park",
            "district": "Colombo",
            "details": "This is one of the best and old place at Colombo for entertainment.",
        })
        assert "District:" not in card
        assert "**Excel World Entertainment Park** · colombo" in card
        assert "Description" not in card
        assert "best and old place" in card

    def test_card_shows_full_description(self):
        long_text = (
            "There are six national parks nearby. "
            "Yala hosts a variety of ecosystems ranging from moist monsoon forests to wetlands."
        )
        card = format_attraction_card({
            "name": "Yala National Park",
            "district": "Hambantota",
            "details": long_text,
        })
        assert "six national parks nearby" in card
        assert "moist monsoon forests" in card
        assert "..." not in card

    def test_itinerary_card_omits_review_snippets(self):
        card = format_attraction_card(
            {
                "name": "Excel World Entertainment Park",
                "district": "Colombo",
                "details": "Excel World Entertainment Park is a theme park in Colombo.",
            },
            include_review=False,
        )
        assert "Visitor feedback" not in card
        assert "Description" not in card
        assert "theme park in Colombo" in card

    def test_description_strips_leading_place_name(self):
        desc = format_details_for_card(
            "Beddagana Wetland Park is a quiet and beautiful place that's perfect for a peaceful day out.",
            name="Beddagana Wetland Park",
        )
        assert "quiet and beautiful" in desc
        assert "Beddagana Wetland Park" not in desc
        assert desc.lower().startswith("a quiet")

    def test_summarize_details_truncates_when_max_len_set(self):
        text = "Word " * 50
        short = summarize_details(text, max_len=40)
        assert short.endswith("...")
        assert len(short) <= 43

    def test_generic_descriptions_are_omitted(self):
        assert format_details_for_card("Very nice place to visit.", name="Batadobalena") == ""
        card = format_attraction_card(
            {"name": "Batadobalena", "district": "Ratnapura", "details": "Very nice place to visit."},
            include_review=False,
        )
        assert "Description:" not in card

    def test_recommendation_response_is_bulleted_list(self):
        text = format_recommendation_response([
            {
                "name": "The Palm Rope Swing",
                "district": "Matara",
                "details": "500 rupees is a bit much for 1 jump, but it's worth it.",
            },
        ])
        assert "District:" not in text
        assert "**The Palm Rope Swing** · Matara" in text
        assert text.strip().startswith("Here are a few options")