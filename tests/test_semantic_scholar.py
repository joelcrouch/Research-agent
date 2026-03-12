import pytest
import httpx
from unittest.mock import MagicMock, patch
from schemas.paper import Paper
from agent.tools.semantic_scholar import get_citation_count

@pytest.fixture
def base_paper():
    return Paper(
        title="Attention is All You Need",
        author=["Vaswani, Ashish"],
        abstract="The dominant sequence transduction models...",
        year=2017,
        arxiv_id="1706.03762",
        pubmed_id=None,
        doi="10.48550/arXiv.1706.03762",
        semantic_scholar_id=None,
        citation_count=None,
        source="arxiv",
        url="https://arxiv.org/abs/1706.03762"
    )

class TestSemanticScholarUnit:
    """Unit tests for get_citation_count using mocks."""

    @patch("httpx.Client.get")
    def test_get_citation_count_by_doi(self, mock_get, base_paper):
        """Test lookup by DOI."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "paperId": "ss_id_123",
            "citationCount": 100,
            "title": base_paper.title,
            "doi": base_paper.doi
        }
        mock_get.return_value = mock_response

        updated_paper = get_citation_count(base_paper)
        
        assert updated_paper.citation_count == 100
        assert updated_paper.semantic_scholar_id == "ss_id_123"
        args, _ = mock_get.call_args
        assert f"DOI:{base_paper.doi}" in args[0]

    @patch("httpx.Client.get")
    def test_get_citation_count_by_arxiv(self, mock_get, base_paper):
        """Test lookup by ArXiv ID with retries (prefix and no prefix)."""
        res_404 = MagicMock(status_code=404)
        res_200 = MagicMock(status_code=200)
        res_200.json.return_value = {"paperId": "ss_arxiv", "citationCount": 50}
        
        # DOI (404), arXiv:ID (404), ID (200)
        mock_get.side_effect = [res_404, res_404, res_200]

        updated_paper = get_citation_count(base_paper)
        assert updated_paper.citation_count == 50
        assert mock_get.call_count == 3

    @patch("httpx.Client.get")
    def test_get_citation_count_by_url(self, mock_get, base_paper):
        """Test lookup by URL fallback."""
        res_404 = MagicMock(status_code=404)
        res_200 = MagicMock(status_code=200)
        res_200.json.return_value = {"paperId": "ss_url", "citationCount": 20}
        
        # DOI, arXiv:ID, ID, URL(200)
        mock_get.side_effect = [res_404, res_404, res_404, res_200]

        updated_paper = get_citation_count(base_paper)
        assert updated_paper.citation_count == 20
        assert mock_get.call_count == 4

    @patch("httpx.Client.get")
    def test_get_citation_count_by_title(self, mock_get, base_paper):
        """Test lookup by title search fallback."""
        res_404 = MagicMock(status_code=404)
        res_title = MagicMock(status_code=200)
        res_title.json.return_value = {"data": [{"paperId": "ss_title", "citationCount": 10}]}
        
        # DOI, arXiv:ID, ID, URL, Title(200)
        mock_get.side_effect = [res_404, res_404, res_404, res_404, res_title]

        updated_paper = get_citation_count(base_paper)
        assert updated_paper.citation_count == 10
        assert mock_get.call_count == 5

    @patch("httpx.Client.get")
    def test_get_citation_count_rate_limit(self, mock_get, base_paper):
        """Test handling of 429 status code."""
        res_429 = MagicMock(status_code=429)
        mock_get.return_value = res_429
        
        updated_paper = get_citation_count(base_paper)
        assert updated_paper.citation_count is None

    @patch("httpx.Client.get")
    def test_get_citation_count_api_error(self, mock_get, base_paper):
        """Test handling of other error status codes (e.g. 500)."""
        res_500 = MagicMock(status_code=500)
        mock_get.return_value = res_500
        
        updated_paper = get_citation_count(base_paper)
        assert updated_paper.citation_count is None

    @patch("httpx.Client.get")
    def test_search_by_title_rate_limit(self, mock_get, base_paper):
        """Test rate limit specifically in title search."""
        res_404 = MagicMock(status_code=404)
        res_429 = MagicMock(status_code=429)
        # DOI, arXiv:ID, ID, URL, Title(429)
        mock_get.side_effect = [res_404, res_404, res_404, res_404, res_429]
        
        updated_paper = get_citation_count(base_paper)
        assert updated_paper.citation_count is None

    @patch("httpx.Client.get")
    def test_search_by_title_exception(self, mock_get, base_paper):
        """Test exception specifically in title search."""
        res_404 = MagicMock(status_code=404)
        mock_get.side_effect = [res_404, res_404, res_404, res_404, Exception("Search Fail")]
        
        updated_paper = get_citation_count(base_paper)
        assert updated_paper.citation_count is None

    @patch("agent.tools.semantic_scholar.httpx.Client")
    def test_get_citation_count_broad_exception(self, mock_client_class, base_paper):
        """Test broad exception handling in get_citation_count."""
        mock_client_class.return_value.__enter__.side_effect = Exception("Fatal")
        updated_paper = get_citation_count(base_paper)
        assert updated_paper == base_paper

    @patch("httpx.Client.get")
    def test_update_paper_populates_missing_doi(self, mock_get, base_paper):
        """Test that DOI is populated if missing from paper but present in result."""
        base_paper.doi = None
        res_200 = MagicMock(status_code=200)
        res_200.json.return_value = {
            "paperId": "ss_id", 
            "citationCount": 5, 
            "doi": "10.1000/new.doi"
        }
        mock_get.return_value = res_200
        
        # Search by ArXiv ID
        updated_paper = get_citation_count(base_paper)
        assert updated_paper.doi == "10.1000/new.doi"

    @patch("httpx.Client.get")
    def test_get_by_id_exception(self, mock_get, base_paper):
        """Verify the catch-all exception block in _get_by_id is hit (lines 77-79)."""
        mock_get.side_effect = Exception("HTTPX Error")
        
        # This will be called for DOI lookup
        updated_paper = get_citation_count(base_paper)
        # Should catch and continue through all ID/URL types and then title search
        # Since all mock_get calls will now raise, it should return original paper
        assert updated_paper.citation_count is None

class TestSemanticScholarIntegration:
    """Integration tests with real API calls."""

    @pytest.mark.integration
    def test_get_citation_count_real_call(self, base_paper):
        """Verify end-to-end lookup with a real paper."""
        updated_paper = get_citation_count(base_paper)
        # Attention is All You Need might be 429'd in CI/shared envs, 
        # but we skip if that happens as per the updated test strategy.
        if updated_paper.citation_count is None:
            pytest.skip("Semantic Scholar rate limit hit during integration test.")
        assert updated_paper.citation_count > 100000
        assert updated_paper.semantic_scholar_id is not None
