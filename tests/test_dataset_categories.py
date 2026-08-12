"""Dataset category stats from repository."""

import pytest

from src.database.postgres import get_session_factory, init_db
from src.database.repositories import AttractionRepository
from src.services.preferences import VALID_CATEGORIES


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()
    yield


class TestDatasetCategories:
    def test_list_category_stats_matches_valid_categories(self):
        session = get_session_factory()()
        try:
            repo = AttractionRepository(session)
            stats, total = repo.list_category_stats()
            assert total >= 0
            names = [row["name"] for row in stats]
            for cat in VALID_CATEGORIES:
                assert cat in names
            assert all(row["count"] >= 0 for row in stats)
        finally:
            session.close()
