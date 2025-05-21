import asyncio
import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import csv
from parser import Company, LLMOutput

from company_lookup import (async_fetch_company_web_info,
                            fetch_company_web_info, parse_llm_response)
from lookup_companies import _industry, generate_final_report
from spreadsheet_parser.analysis import run_async


class TestFetchCompanyWebInfo(unittest.TestCase):
    @patch("company_lookup.openai.OpenAI")
    def test_prompt_includes_csv_details(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_client.responses.create.return_value = type(
            "Resp",
            (),
            {"output_text": "ok"},
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
                        industries=["Manufacturing"],
                        headquarters_location="New York, NY",
                        description="Leading gadget manufacturer",
                        cb_rank="1",
                    )

                    fetch_company_web_info(company, model="test-model")

                    args, kwargs = mock_client.responses.create.call_args
                    self.assertEqual(kwargs["model"], "test-model")
                    self.assertEqual(
                        kwargs["tools"], [{"type": "web_search_preview"}]
                    )
                    user_content = kwargs["input"]
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
        self.assertIn("is_possibly_malformed", user_content)
        self.assertIn("malformation_reason", user_content)
        self.assertIn("business_model_summary", user_content)

    def test_parse_llm_response(self):
        text = (
            "Acme summary.\n"
            "```json\n"
            '{"organization_name": "Acme", "supportive": 0.75, "sub_category": "Generative AI", "is_business": true, "is_possibly_malformed": false, "business_model_summary": "Acme summary"}'
            "\n```"
        )
        result = parse_llm_response(text)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, LLMOutput)
        self.assertAlmostEqual(result.supportive, 0.75)
        self.assertEqual(result.raw.get("organization_name"), "Acme")
        self.assertEqual(result.sub_category, "Generative AI")
        self.assertTrue(result.is_business)
        self.assertEqual(result.business_model_summary, "Acme summary")
        self.assertFalse(result.is_possibly_malformed)
        self.assertIsNone(result.malformation_reason)

    def test_parse_llm_response_edge_cases(self):
        self.assertIsNone(parse_llm_response("nonsense"))

        bad_json = "```json\n{bad}\n```"
        self.assertIsNone(parse_llm_response(bad_json))

        text_yes = '```json\n{"supportive": "yes", "is_business": true}\n```'
        self.assertEqual(parse_llm_response(text_yes).supportive, 1.0)

        text_no = '```json\n{"supportive": "no", "is_business": true}\n```'
        self.assertEqual(parse_llm_response(text_no).supportive, 0.0)

        text_str_number = '```json\n{"supportive": "0.25", "is_business": true}\n```'
        self.assertAlmostEqual(parse_llm_response(text_str_number).supportive, 0.25)

        text_out_of_range = '```json\n{"supportive": 1.5, "is_business": true}\n```'
        result = parse_llm_response(text_out_of_range)
        self.assertIsNotNone(result)
        self.assertIsNone(result.supportive)

        bool_yes = '```json\n{"supportive": 0.5, "is_business": "yes"}\n```'
        self.assertTrue(parse_llm_response(bool_yes).is_business)

        bool_no = '```json\n{"supportive": 0.5, "is_business": "no"}\n```'
        self.assertFalse(parse_llm_response(bool_no).is_business)

    def test_parse_llm_response_malformation_reason(self):
        text = (
            "Intro.\n"
            "```json\n"
            '{"supportive": 0.4, "is_business": true, "is_possibly_malformed": true, "malformation_reason": "bad header"}'
            "\n```"
        )
        result = parse_llm_response(text)
        self.assertTrue(result.is_possibly_malformed)
        self.assertEqual(result.malformation_reason, "bad header")

    def test_parse_llm_response_reason_cleared_when_flag_false(self):
        text = (
            "Intro.\n"
            "```json\n"
            '{"supportive": 0.7, "is_business": true, "is_possibly_malformed": false, "malformation_reason": "bad header"}'
            "\n```"
        )
        result = parse_llm_response(text)
        self.assertFalse(result.is_possibly_malformed)
        self.assertIsNone(result.malformation_reason)

    def test_parse_llm_response_no_label(self):
        text = "Intro.\n" "```\n" '{"supportive": 0.6, "is_business": true}' "\n```"
        result = parse_llm_response(text)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, LLMOutput)
        self.assertAlmostEqual(result.supportive, 0.6)

    def test_parse_llm_response_single_quotes(self):
        text = "Intro.\n" "```json\n" "{'supportive': '0.3', 'is_business': True}" "\n```"
        result = parse_llm_response(text)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, LLMOutput)
        self.assertAlmostEqual(result.supportive, 0.3)

    def test_parse_llm_response_embedded_apostrophes(self):
        text = (
            "Intro.\n"
            "```json\n"
            '{"organization_name": "O\'Reilly", "supportive": 0.6, "is_business": True}'
            "\n```"
        )
        result = parse_llm_response(text)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, LLMOutput)
        self.assertEqual(result.raw.get("organization_name"), "O'Reilly")
        self.assertAlmostEqual(result.supportive, 0.6)

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
        mock_client.responses.create.return_value = type(
            "Resp",
            (),
            {"output_text": "ok"},
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

        self.assertEqual(mock_client.responses.create.call_count, 1)

    @patch("company_lookup.openai.AsyncOpenAI", create=True)
    def test_async_cache_reused_for_same_seed(self, mock_async_openai):
        mock_client = mock_async_openai.return_value
        mock_client.aclose = AsyncMock()

        async def fake_create(**kwargs):
            return type("Resp", (), {"output_text": "ok"})

        mock_client.responses.create = AsyncMock(side_effect=fake_create)

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

        self.assertEqual(mock_client.responses.create.call_count, 1)

    @patch("company_lookup.openai.OpenAI")
    def test_cache_not_reused_for_different_model(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_client.responses.create.return_value = type(
            "Resp",
            (),
            {"output_text": "ok"},
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
                    fetch_company_web_info("Acme Corp", model="model-a")
                    fetch_company_web_info("Acme Corp", model="model-b")

        self.assertEqual(mock_client.responses.create.call_count, 2)

    @patch("company_lookup.openai.AsyncOpenAI", create=True)
    def test_async_cache_not_reused_for_different_model(self, mock_async_openai):
        mock_client = mock_async_openai.return_value
        mock_client.aclose = AsyncMock()

        async def fake_create(**kwargs):
            return type("Resp", (), {"output_text": "ok"})

        mock_client.responses.create = AsyncMock(side_effect=fake_create)

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
                        async_fetch_company_web_info("Acme Corp", model="model-a")
                    )
                    asyncio.run(
                        async_fetch_company_web_info("Acme Corp", model="model-b")
                    )

        self.assertEqual(mock_client.responses.create.call_count, 2)


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
                industries=["Manufacturing"],
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
                industries=["Technology"],
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
                industries=["Software"],
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
        report = generate_final_report(
            companies,
            stances,
            justifications=justifications,
            is_malformed_flags=[False, False, True],
        )
        # Industry summary lines should not be present
        self.assertNotIn("supportive company found", report)
        self.assertIn("Overall 1/2 companies are supportive", report)
        self.assertIn("Supportive companies by industry", report)
        self.assertIn("  Manufacturing: #################### (1/1)", report)
        self.assertNotIn("  Software: ########## (1/1)", report)
        self.assertIn("  Technology:  (0/1)", report)
        self.assertIn("Average stance per industry", report)
        self.assertIn("Top companies: Acme Corp, Globex Inc", report)
        self.assertIn("Supportive companies tend to be smaller", report)
        self.assertIn("Support by IPO status", report)
        self.assertIn("Support by revenue range", report)
        self.assertIn("Support by CB rank", report)
        self.assertIn("Average company size metric", report)
        self.assertIn("Input data statistics", report)
        self.assertIn("Most common industry: Manufacturing (1)", report)
        self.assertIn("Most common IPO status: Unknown (2)", report)
        self.assertIn("Employee counts (min/median/max): 175 / 462 / 750", report)
        self.assertIn("Conclusions:", report)
        self.assertIn("Example justifications:", report)
        self.assertIn("Acme Corp (Support): Because open standards are good", report)
        self.assertIn("Possibly malformed datapoints: 1", report)

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
                industries=["Manufacturing"],
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
                industries=["Technology"],
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
    def _make_company(self, industries: str | list[str]) -> Company:
        if isinstance(industries, str):
            industries = [industries]
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


class TestIndustryLimit(unittest.TestCase):
    def test_top_25_only(self):
        companies = []
        stances = []
        letters = [chr(ord('A') + i) for i in range(26)]
        for letter in letters:
            companies.append(
                Company(
                    organization_name=f"Co{letter}",
                    organization_name_url=None,
                    estimated_revenue_range=None,
                    ipo_status=None,
                    operating_status=None,
                    acquisition_status=None,
                    company_type=None,
                    number_of_employees=None,
                    full_description=None,
                    industries=[f"Industry{letter}"],
                    headquarters_location=None,
                    description=None,
                    cb_rank=None,
                )
            )
            stances.append(0.9)
        companies.append(
            Company(
                organization_name="Extra",
                organization_name_url=None,
                estimated_revenue_range=None,
                ipo_status=None,
                operating_status=None,
                acquisition_status=None,
                company_type=None,
                number_of_employees=None,
                full_description=None,
                industries=["IndustryA"],
                headquarters_location=None,
                description=None,
                cb_rank=None,
            )
        )
        stances.append(0.9)

        report = generate_final_report(companies, stances)
        self.assertNotIn("supportive company found", report)


class TestRunAsync(unittest.TestCase):
    @patch("spreadsheet_parser.analysis._sample_data_quality_report", new_callable=AsyncMock)
    @patch("spreadsheet_parser.analysis.async_report_to_abstract", new_callable=AsyncMock)
    @patch("company_lookup.async_fetch_company_web_info")
    def test_qualitative_justification_column(self, mock_fetch, mock_abstract, mock_quality):
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

        async def fake_fetch(name, *, return_cache_info=False, model=None, client=None):
            return (responses[name], False)

        mock_fetch.side_effect = fake_fetch
        mock_abstract.return_value = "Abstract"
        mock_quality.return_value = "- ok"

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
                industries=["Technology"],
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
                industries=["Manufacturing"],
                headquarters_location=None,
                description=None,
                cb_rank=None,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                asyncio.run(run_async(companies, 1, pathlib.Path(tmpdir)))
            csv_path = pathlib.Path(tmpdir) / "company_analysis.csv"
            dq_path = pathlib.Path(tmpdir) / "data_quality_report.csv"
            dq_txt_path = pathlib.Path(tmpdir) / "data_quality_report.txt"
            abstract_path = pathlib.Path(tmpdir) / "abstract.txt"
            with csv_path.open(newline="") as f:
                rows = list(csv.reader(f))
            self.assertTrue(abstract_path.exists())
            self.assertTrue(dq_path.exists())
            self.assertTrue(dq_txt_path.exists())
            with dq_path.open(newline="") as f:
                dq_rows = list(csv.reader(f))
            dq_text = dq_txt_path.read_text().strip()

        self.assertEqual(rows[0][0], "Company Name")
        acme = rows[1]
        globex = rows[2]
        self.assertNotEqual(acme[3], acme[5])
        self.assertEqual(globex[3], globex[5])
        self.assertEqual(dq_rows[0], ["Observation"])
        self.assertEqual(dq_rows[1][0], "ok")
        self.assertEqual(dq_text, "- ok")

    @patch("spreadsheet_parser.analysis._sample_data_quality_report", new_callable=AsyncMock)
    @patch("spreadsheet_parser.analysis.async_report_to_abstract", new_callable=AsyncMock)
    @patch("company_lookup.async_fetch_company_web_info")
    def test_quality_csv_lists_malformed(self, mock_fetch, mock_abstract, mock_quality):
        responses = {
            "Acme Corp": (
                "Summary one.\n"
                "```json\n"
                '{"supportive": 0.9, "is_business": true}'
                "\n```"
            ),
            "Globex Inc": (
                "Summary two.\n"
                "```json\n"
                '{"supportive": 0.4, "is_business": true, "is_possibly_malformed": true}'
                "\n```"
            ),
        }

        async def fake_fetch(name, *, return_cache_info=False, model=None, client=None):
            return (responses[name], False)

        mock_fetch.side_effect = fake_fetch
        mock_abstract.return_value = "Abstract"
        mock_quality.return_value = "- note"

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
                industries=["Technology"],
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
                industries=["Manufacturing"],
                headquarters_location=None,
                description=None,
                cb_rank=None,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                asyncio.run(run_async(companies, 1, pathlib.Path(tmpdir)))
            dq_path = pathlib.Path(tmpdir) / "data_quality_report.csv"
            with dq_path.open(newline="") as f:
                dq_rows = list(csv.reader(f))

        self.assertEqual(dq_rows[0], ["Observation"])
        self.assertIn("Malformed Data", dq_rows[3])
        self.assertIn("Globex Inc", dq_rows[-1])

    @patch("spreadsheet_parser.analysis._sample_data_quality_report", new_callable=AsyncMock)
    @patch("spreadsheet_parser.analysis.async_report_to_abstract", new_callable=AsyncMock)
    @patch("company_lookup.async_fetch_company_web_info")
    def test_malformed_data_csv(self, mock_fetch, mock_abstract, mock_quality):
        responses = {
            "Acme Corp": (
                "Summary one.\n"
                "```json\n"
                '{"supportive": 0.9, "is_business": true}'
                "\n```"
            ),
            "Globex Inc": (
                "Summary two.\n"
                "```json\n"
                '{"supportive": 0.4, "is_business": true, "is_possibly_malformed": true, "malformation_reason": "bad"}'
                "\n```"
            ),
        }

        async def fake_fetch(name, *, return_cache_info=False, model=None, client=None):
            return (responses[name], False)

        mock_fetch.side_effect = fake_fetch
        mock_abstract.return_value = "Abstract"
        mock_quality.return_value = "- note"

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
                industries=["Technology"],
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
                industries=["Manufacturing"],
                headquarters_location=None,
                description=None,
                cb_rank=None,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                asyncio.run(run_async(companies, 1, pathlib.Path(tmpdir)))
            mal_path = pathlib.Path(tmpdir) / "malformed_data.csv"
            self.assertTrue(mal_path.exists())
            with mal_path.open(newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["organization_name"], "Globex Inc")
        self.assertEqual(row.get("malformation_reason"), "bad")


if __name__ == "__main__":
    unittest.main()
