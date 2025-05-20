import sys
import types
import unittest
import os
from unittest.mock import patch, AsyncMock, MagicMock

# Provide a minimal openai stub if the real package is unavailable
if "openai" not in sys.modules:

    def make_client(*args, **kwargs):
        return types.SimpleNamespace(
            responses=types.SimpleNamespace(create=lambda **kwargs: None)
        )

    openai_stub = types.SimpleNamespace(OpenAI=make_client)
    sys.modules["openai"] = openai_stub

from datetime import datetime
from parser import Company
from spreadsheet_parser.analysis import _cb_rank_value, _employee_count, percentile_ranks
from spreadsheet_parser.llm import report_to_abstract


class TestCBRankValue(unittest.TestCase):
    def test_handles_numeric_input(self):
        self.assertEqual(_cb_rank_value(123), 123)
        self.assertEqual(_cb_rank_value(456.0), 456)

    def test_handles_strings(self):
        self.assertEqual(_cb_rank_value("1,234"), 1234)
        self.assertIsNone(_cb_rank_value(None))


class TestEmployeeCount(unittest.TestCase):
    def make_company(self, value):
        return Company(
            organization_name="Test",
            organization_name_url=None,
            estimated_revenue_range=None,
            ipo_status=None,
            operating_status=None,
            acquisition_status=None,
            company_type=None,
            number_of_employees=value,
            full_description=None,
            industries=None,
            headquarters_location=None,
            description=None,
            cb_rank=None,
        )

    def test_numeric_and_range(self):
        self.assertEqual(_employee_count(self.make_company("5-10")), 7.5)
        self.assertEqual(_employee_count(self.make_company(42)), 42.0)

    def test_datetime_returns_none(self):
        self.assertIsNone(_employee_count(self.make_company(datetime(2020, 1, 1))))


class TestReportToAbstract(unittest.TestCase):
    @patch("spreadsheet_parser.llm.openai.AsyncOpenAI", create=True)
    @patch("spreadsheet_parser.llm.openai.OpenAI")
    def test_prompt_contains_report(self, mock_openai, mock_async):
        response_obj = type("Resp", (), {"output_text": "abs"})
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=response_obj)
        mock_async.return_value = mock_client

        with patch.dict(os.environ, {"OPENAI_API_KEY": "x"}):
            result = report_to_abstract("Final Report text", model="test-model")

        args, kwargs = mock_client.responses.create.call_args
        self.assertEqual(kwargs["model"], "test-model")
        self.assertIn("Final Report text", kwargs["input"])
        self.assertEqual(result, "abs")


class TestPercentileRanks(unittest.TestCase):
    def test_basic_percentiles(self):
        vals = [10, 20, 30]
        ranks = percentile_ranks(vals)
        self.assertAlmostEqual(ranks[0], 0.1666, places=2)
        self.assertAlmostEqual(ranks[1], 0.5, places=2)
        self.assertAlmostEqual(ranks[2], 0.8333, places=2)

    def test_handles_ties_and_none(self):
        vals = [5, None, 10, 5]
        ranks = percentile_ranks(vals)
        self.assertIsNone(ranks[1])
        self.assertAlmostEqual(ranks[0], ranks[3], places=3)
        self.assertAlmostEqual(ranks[2], 0.8333, places=3)


if __name__ == "__main__":
    unittest.main()
