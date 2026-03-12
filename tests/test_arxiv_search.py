import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, UTC
from agent.tools.arxiv_search import search_arxiv
from schemas.paper import Paper

class TestArxivSearchUnit:
    """Unit tests for search_arxiv using mocks."""

    @patch("arxiv.Client.results")
    def test_search_arxiv_mapping(self, mock_results):
        """Verify that arxiv.Result fields are correctly mapped to Paper schema."""
        # Create a mock arxiv.Result object
        mock_result = MagicMock()
        mock_result.title = "Test Paper"
        mock_result.authors = [MagicMock()]
        mock_result.authors[0].name = "Author One"
        mock_result.summary = "This is a test abstract."
        mock_result.published = datetime(2024, 1, 1, tzinfo=UTC)
        mock_result.get_short_id.return_value = "2401.12345"
        mock_result.doi = "10.1234/test.doi"
        mock_result.entry_id = "http://arxiv.org/abs/2401.12345v1"

        # Mock the results generator
        mock_results.return_value = [mock_result]

        papers = search_arxiv("test query", max_results=1)

        assert len(papers) == 1
        paper = papers[0]
        assert isinstance(paper, Paper)
        assert paper.title == "Test Paper"
        assert paper.author == ["Author One"]
        assert paper.abstract == "This is a test abstract."
        assert paper.year == 2024
        assert paper.arxiv_id == "2401.12345"
        assert paper.source == "arxiv"
        assert paper.url == "http://arxiv.org/abs/2401.12345v1"

    @patch("arxiv.Client.results")
    def test_search_arxiv_error_handling(self, mock_results):
        """Verify that exceptions are caught and return an empty list."""
        mock_results.side_effect = Exception("API Failure")
        
        papers = search_arxiv("test query")
        assert papers == []

    @patch("arxiv.Client.results")
    def test_search_arxiv_date_filtering(self, mock_results):
        """Verify that local date filtering works correctly."""
        # Result 1: Old
        res1 = MagicMock()
        res1.published = datetime(2020, 1, 1, tzinfo=UTC)
        
        # Result 2: New
        res2 = MagicMock()
        res2.published = datetime(2024, 1, 1, tzinfo=UTC)
        res2.title = "New Paper"
        res2.authors = []
        res2.get_short_id.return_value = "id"
        res2.doi = None
        res2.entry_id = "url"
        res2.summary = "summary"

        mock_results.return_value = [res1, res2]

        # Search for papers from 2023 onwards
        papers = search_arxiv("test", date_from="2023-01-01")
        
        assert len(papers) == 1
        assert papers[0].title == "New Paper"

    def test_search_arxiv_empty_query(self):
        """Verify that an empty query returns an empty list immediately."""
        assert search_arxiv("") == []
        assert search_arxiv("   ") == []

    @patch("arxiv.Client.results")
    def test_search_arxiv_malformed_date(self, mock_results):
        """Verify that a malformed date logs a warning but doesn't crash (ignores filter)."""
        res1 = MagicMock()
        res1.published = datetime(2024, 1, 1, tzinfo=UTC)
        res1.title = "Test Paper"
        res1.authors = []
        res1.get_short_id.return_value = "id"
        res1.doi = None
        res1.entry_id = "url"
        res1.summary = "summary"
        
        mock_results.return_value = [res1]
        
        # Should ignore the invalid date filter and return the paper
        papers = search_arxiv("test", date_from="not-a-date")
        assert len(papers) == 1
        assert papers[0].title == "Test Paper"

    @patch("arxiv.Client.results")
    def test_search_arxiv_date_to_filtering(self, mock_results):
        """Verify that date_to filtering works correctly."""
        res1 = MagicMock()
        res1.published = datetime(2024, 1, 1, tzinfo=UTC)
        res1.title = "New Paper"
        res1.authors = []
        res1.get_short_id.return_value = "id1"
        res1.doi = None
        res1.entry_id = "url1"
        res1.summary = "summary1"

        res2 = MagicMock()
        res2.published = datetime(2020, 1, 1, tzinfo=UTC)
        res2.title = "Old Paper"
        res2.authors = []
        res2.get_short_id.return_value = "id2"
        res2.doi = None
        res2.entry_id = "url2"
        res2.summary = "summary2"

        mock_results.return_value = [res1, res2]

        # Only old paper should remain
        papers = search_arxiv("test", date_to="2021-01-01")
        assert len(papers) == 1
        assert papers[0].title == "Old Paper"

    @patch("arxiv.Client.results")
    def test_search_arxiv_no_results_from_api(self, mock_results):
        """Verify handling when the API returns an empty iterator."""
        mock_results.return_value = iter([])
        papers = search_arxiv("nonexistent topic")
        assert papers == []

    @patch("arxiv.Client.results")
    def test_search_arxiv_filtered_to_zero(self, mock_results):
        """Verify handling when results exist but are all filtered out."""
        res1 = MagicMock()
        res1.published = datetime(2020, 1, 1, tzinfo=UTC)
        mock_results.return_value = [res1]

        # Filter for only 2024 papers
        papers = search_arxiv("test", date_from="2024-01-01")
        assert papers == []

class TestArxivSearchIntegration:
    """Integration tests with real API calls."""

    @pytest.mark.integration
    def test_search_arxiv_real_call(self):
        """Test a real ArXiv search to verify end-to-end functionality."""
        papers = search_arxiv("long-term potentiation", max_results=2)
        
        assert len(papers) > 0
        assert all(isinstance(p, Paper) for p in papers)
        assert all(p.source == "arxiv" for p in papers)
        # Check that we got something relevant
        found_match = False
        for p in papers:
            if "potentiation" in p.title.lower() or (p.abstract and "potentiation" in p.abstract.lower()):
                found_match = True
                break
        assert found_match
