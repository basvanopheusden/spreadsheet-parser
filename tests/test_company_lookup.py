import asyncio
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, patch

# Provide a minimal openai stub if the real package is unavailable
if "openai" not in sys.modules:

    def make_client(*args, **kwargs):
        return types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kwargs: None)
            )
        )

    openai_stub = types.SimpleNamespace(
        ChatCompletion=types.SimpleNamespace(create=lambda **kwargs: None),
        OpenAI=make_client,
    )
    sys.modules["openai"] = openai_stub

import csv
from parser import Company

from company_lookup import (async_fetch_company_web_info,
                            fetch_company_web_info, parse_llm_response)
from lookup_companies import _industry, generate_final_report
from spreadsheet_parser.analysis import run_async


class TestFetchCompanyWebInfo(unittest.TestCase):
    @patch("company_lookup.openai.OpenAI")
    def test_prompt_includes_csv_details(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = type(
            "Resp",
            (),
            {"choices": [type("Choice", (), {"message": {"content": "ok"}})]},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=pathlib.Path(tmpdir)):
                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                    company = Company(
                        organization_name="Acme Corp",
                        organization_name_url="https://acme.example.com",
                        estimated_revenue_range="$10M-$50M",
                        ipo_status="Private",
                        operating_status="Operating",
                        acquisition_status=None,
                        company_type="For Profit",
                        number_of_employees="100-250",
                        full_description="Acme Corp specializes in gadgets and gizmos.",
                        industries="Manufacturing",
                        headquarters_location="New York, NY",
                        description="Leading gadget manufacturer",
                        cb_rank="1",
                    )

                    fetch_company_web_info(company, model="test-model")

                    args, kwargs = mock_client.chat.completions.create.call_args
                    self.assertEqual(kwargs["model"], "test-model")
                    user_content = kwargs["messages"][1]["content"]
                    self.assertIn("Acme Corp", user_content)
                    self.assertIn(
                        '"estimated_revenue_range": "$10M-$50M"', user_content
                    )
                    self.assertIn(
                        '"headquarters_location": "New York, NY"', user_content
                    )
                    self.assertIn("'supportive'", user_content)
                    self.assertIn(
                        "scale from 0 (strong opponent) to 1 (strong proponent)",
                        user_content,
                    )
                    self.assertIn("Mozilla", user_content)
                    self.assertIn("Palantir", user_content)
        self.assertIn("Return ONLY a JSON code block", user_content)
        self.assertIn("```json", user_content)
        self.assertIn("sub_category", user_content)
        self.assertIn("justification", user_content)
        self.assertIn("is_business", user_content)

    def test_parse_llm_response(self):
        text = (
            "Acme summary.\n"
            "```json\n"
            '{"organization_name": "Acme", "supportive": 0.75, "sub_category": "Generative AI", "is_business": true}'
            "\n```"
        )
        result = parse_llm_response(text)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertAlmostEqual(result.get("supportive"), 0.75)
        self.assertEqual(result.get("organization_name"), "Acme")
        self.assertEqual(result.get("sub_category"), "Generative AI")
        self.assertTrue(result.get("is_business"))

    def test_parse_llm_response_edge_cases(self):
        self.assertIsNone(parse_llm_response("nonsense"))

        bad_json = "```json\n{bad}\n```"
        self.assertIsNone(parse_llm_response(bad_json))

        text_yes = '```json\n{"supportive": "yes", "is_business": true}\n```'
        self.assertEqual(parse_llm_response(text_yes)["supportive"], 1.0)

        text_no = '```json\n{"supportive": "no", "is_business": true}\n```'
        self.assertEqual(parse_llm_response(text_no)["supportive"], 0.0)

        text_str_number = '```json\n{"supportive": "0.25", "is_business": true}\n```'
        self.assertAlmostEqual(parse_llm_response(text_str_number)["supportive"], 0.25)

        text_out_of_range = '```json\n{"supportive": 1.5, "is_business": true}\n```'
        result = parse_llm_response(text_out_of_range)
        self.assertIsNotNone(result)
        self.assertIsNone(result.get("supportive"))

        bool_yes = '```json\n{"supportive": 0.5, "is_business": "yes"}\n```'
        self.assertTrue(parse_llm_response(bool_yes)["is_business"])

        bool_no = '```json\n{"supportive": 0.5, "is_business": "no"}\n```'
        self.assertFalse(parse_llm_response(bool_no)["is_business"])

    def test_parse_llm_response_no_label(self):
        text = "Intro.\n" "```\n" '{"supportive": 0.6, "is_business": true}' "\n```"
        result = parse_llm_response(text)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertAlmostEqual(result.get("supportive"), 0.6)

    def test_parse_llm_response_single_quotes(self):
        text = "Intro.\n" "```json\n" "{'supportive': '0.3', 'is_business': True}" "\n```"
        result = parse_llm_response(text)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertAlmostEqual(result.get("supportive"), 0.3)

    def test_parse_llm_response_embedded_apostrophes(self):
        text = (
            "Intro.\n"
            "```json\n"
            '{"organization_name": "O\'Reilly", "supportive": 0.6, "is_business": True}'
            "\n```"
        )
        result = parse_llm_response(text)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("organization_name"), "O'Reilly")
        self.assertAlmostEqual(result.get("supportive"), 0.6)

    def test_parse_llm_response_missing_keys(self):
        only_support = '```json\n{"supportive": 0.5}\n```'
        self.assertIsNone(parse_llm_response(only_support))

        only_business = '```json\n{"is_business": True}\n```'
        self.assertIsNone(parse_llm_response(only_business))

    def test_parse_llm_response_missing_keys_raise(self):
        text = '```json\n{"supportive": 0.5}\n```'
        with self.assertRaises(KeyError):
            parse_llm_response(text, raise_on_missing=True)

    @patch("company_lookup.openai.OpenAI")
    def test_cache_reused_for_same_seed(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = type(
            "Resp",
            (),
            {"choices": [type("Choice", (), {"message": {"content": "ok"}})]},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=pathlib.Path(tmpdir)):
                with patch.dict(
                    os.environ,
                    {
                        "OPENAI_API_KEY": "test-key",
                        "OPENAI_SEED": "123",
                    },
                ):
                    fetch_company_web_info("Acme Corp", model="test-model")
                    fetch_company_web_info("Acme Corp", model="test-model")

        self.assertEqual(mock_client.chat.completions.create.call_count, 1)

    @patch("company_lookup.openai.AsyncOpenAI", create=True)
    def test_async_cache_reused_for_same_seed(self, mock_async_openai):
        mock_client = mock_async_openai.return_value

        async def fake_create(**kwargs):
            return type(
                "Resp",
                (),
                {"choices": [type("Choice", (), {"message": {"content": "ok"}})]},
            )

        mock_client.chat.completions.create = AsyncMock(side_effect=fake_create)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=pathlib.Path(tmpdir)):
                with patch.dict(
                    os.environ,
                    {
                        "OPENAI_API_KEY": "test-key",
                        "OPENAI_SEED": "123",
                    },
                ):
                    asyncio.run(
                        async_fetch_company_web_info("Acme Corp", model="test-model")
                    )
                    asyncio.run(
                        async_fetch_company_web_info("Acme Corp", model="test-model")
                    )

        self.assertEqual(mock_client.chat.completions.create.call_count, 1)


class TestFinalReport(unittest.TestCase):
    def test_generate_final_report(self):
        companies = [
            Company(
                organization_name="Acme Corp",
                organization_name_url=None,
                estimated_revenue_range=None,
                ipo_status=None,
                operating_status=None,
                acquisition_status=None,
                company_type=None,
                number_of_employees="100-250",
                full_description=None,
                industries="Manufacturing",
                headquarters_location=None,
                description=None,
                cb_rank=None,
            ),
            Company(
                organization_name="Globex Inc",
                organization_name_url=None,
                estimated_revenue_range=None,
                ipo_status=None,
                operating_status=None,
                acquisition_status=None,
                company_type=None,
                number_of_employees="500-1000",
                full_description=None,
                industries="Technology",
                headquarters_location=None,
                description=None,
                cb_rank=None,
            ),
            Company(
                organization_name="Initech",
                organization_name_url=None,
                estimated_revenue_range=None,
                ipo_status=None,
                operating_status=None,
                acquisition_status=None,
                company_type=None,
                number_of_employees="50-100",
                full_description=None,
                industries="Software",
                headquarters_location=None,
                description=None,
                cb_rank=None,
            ),
        ]

        stances = [0.8, 0.4, 0.9]
        justifications = [
            "Because open standards are good",
            "Less interested in openness",
            "Strong advocate for open APIs",
        ]
        report = generate_final_report(companies, stances, justifications=justifications)
        self.assertIn("Manufacturing: supportive company found", report)
        self.assertIn("Technology: no supportive company found", report)
        self.assertIn("Software: supportive company found", report)
        self.assertIn("Overall 2/3 companies are supportive", report)
        self.assertIn("Supportive companies by industry", report)
        self.assertIn("  Manufacturing: ########## (1/1)", report)
        self.assertIn("  Software: ########## (1/1)", report)
        self.assertIn("  Technology:  (0/1)", report)
        self.assertIn("Average stance per industry", report)
        self.assertIn("Top companies: Initech, Acme Corp, Globex Inc", report)
        self.assertIn("Supportive companies tend to be smaller", report)
        self.assertIn("Support by IPO status", report)
        self.assertIn("Support by revenue range", report)
        self.assertIn("Support by CB rank", report)
        self.assertIn("Average company size metric", report)
        self.assertIn("Input data statistics", report)
        self.assertIn("Most common industry: Manufacturing (1)", report)
        self.assertIn("Most common IPO status: Unknown (3)", report)
        self.assertIn("Employee counts (min/median/max): 75 / 175 / 750", report)
        self.assertIn("Conclusions:", report)
        self.assertIn("Example justifications:", report)
        self.assertIn("Acme Corp (Support): Because open standards are good", report)

    def test_excludes_non_business(self):
        companies = [
            Company(
                organization_name="Acme Corp",
                organization_name_url=None,
                estimated_revenue_range=None,
                ipo_status=None,
                operating_status=None,
                acquisition_status=None,
                company_type=None,
                number_of_employees="100-250",
                full_description=None,
                industries="Manufacturing",
                headquarters_location=None,
                description=None,
                cb_rank=None,
            ),
            Company(
                organization_name="Globex Institute",
                organization_name_url=None,
                estimated_revenue_range=None,
                ipo_status=None,
                operating_status=None,
                acquisition_status=None,
                company_type=None,
                number_of_employees="500-1000",
                full_description=None,
                industries="Technology",
                headquarters_location=None,
                description=None,
                cb_rank=None,
            ),
        ]

        stances = [0.8, 0.4]
        justifications = [
            "Because open standards are good",
            "Less interested in openness",
        ]
        is_biz = [True, False]
        report = generate_final_report(
            companies,
            stances,
            justifications=justifications,
            is_business_flags=is_biz,
        )
        self.assertIn("Overall 1/1 companies are supportive", report)
        self.assertNotIn("Globex Institute", report)


class TestIndustryNormalization(unittest.TestCase):
    def _make_company(self, industries: str) -> Company:
        return Company(
            organization_name="Dummy",
            organization_name_url=None,
            estimated_revenue_range=None,
            ipo_status=None,
            operating_status=None,
            acquisition_status=None,
            company_type=None,
            number_of_employees=None,
            full_description=None,
            industries=industries,
            headquarters_location=None,
            description=None,
            cb_rank=None,
        )

    def test_aliases_map_to_canonical(self):
        base = self._make_company("Artificial Intelligence")
        alias1 = self._make_company("AI")
        alias2 = self._make_company("Artificial Intelligence (AI)")

        self.assertEqual(_industry(base), "Artificial Intelligence")
        self.assertEqual(_industry(alias1), "Artificial Intelligence")
        self.assertEqual(_industry(alias2), "Artificial Intelligence")


class TestRunAsync(unittest.TestCase):
    @patch("lookup_companies.async_fetch_company_web_info")
    def test_qualitative_justification_column(self, mock_fetch):
        responses = {
            "Acme Corp": (
                "Summary one.\n"
                "```json\n"
                '{"supportive": 0.9, "is_business": true, "justification": "Because open standards are good"}'
                "\n```"
            ),
            "Globex Inc": (
                "Summary two.\n"
                "```json\n"
                '{"supportive": 0.4, "is_business": true}'
                "\n```"
            ),
        }

        async def fake_fetch(name, *, return_cache_info=False):
            return (responses[name], False)

        mock_fetch.side_effect = fake_fetch

        companies = [
            Company(
                organization_name="Acme Corp",
                organization_name_url=None,
                estimated_revenue_range=None,
                ipo_status=None,
                operating_status=None,
                acquisition_status=None,
                company_type=None,
                number_of_employees=None,
                full_description=None,
                industries="Technology",
                headquarters_location=None,
                description=None,
                cb_rank=None,
            ),
            Company(
                organization_name="Globex Inc",
                organization_name_url=None,
                estimated_revenue_range=None,
                ipo_status=None,
                operating_status=None,
                acquisition_status=None,
                company_type=None,
                number_of_employees=None,
                full_description=None,
                industries="Manufacturing",
                headquarters_location=None,
                description=None,
                cb_rank=None,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            asyncio.run(run_async(companies, 1, pathlib.Path(tmpdir)))
            csv_path = pathlib.Path(tmpdir) / "company_analysis.csv"
            with csv_path.open(newline="") as f:
                rows = list(csv.reader(f))

        self.assertEqual(rows[0][0], "Company Name")
        acme = rows[1]
        globex = rows[2]
        self.assertNotEqual(acme[3], acme[5])
        self.assertEqual(globex[3], globex[5])


if __name__ == "__main__":
    unittest.main()
