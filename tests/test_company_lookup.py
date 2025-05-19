import os
import sys
import types
import unittest
import tempfile
import pathlib
import asyncio
from unittest.mock import patch, AsyncMock

# Provide a minimal openai stub if the real package is unavailable
if 'openai' not in sys.modules:
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
    sys.modules['openai'] = openai_stub

from parser import Company
from company_lookup import (
    fetch_company_web_info,
    async_fetch_company_web_info,
    parse_llm_response,
)
from lookup_companies import generate_final_report, _industry


class TestFetchCompanyWebInfo(unittest.TestCase):
    @patch('company_lookup.openai.OpenAI')
    def test_prompt_includes_csv_details(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = type(
            'Resp',
            (),
            {'choices': [type('Choice', (), {'message': {'content': 'ok'}})]},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pathlib.Path.home', return_value=pathlib.Path(tmpdir)):
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
                    self.assertEqual(kwargs['model'], 'test-model')
                    user_content = kwargs['messages'][1]['content']
                    self.assertIn("Acme Corp", user_content)
                    self.assertIn('"estimated_revenue_range": "$10M-$50M"', user_content)
                    self.assertIn('"headquarters_location": "New York, NY"', user_content)
                    self.assertIn("'supportive'", user_content)
                    self.assertIn("scale from 0 (strong opponent) to 1 (strong proponent)", user_content)
                    self.assertIn("Mozilla", user_content)
                    self.assertIn("Palantir", user_content)
                    self.assertIn("Return ONLY a JSON code block", user_content)
                    self.assertIn("```json", user_content)

    def test_parse_llm_response(self):
        text = (
            "Acme summary.\n"
            "```json\n"
            '{"organization_name": "Acme", "supportive": 0.75}'
            "\n```"
        )
        result = parse_llm_response(text)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertAlmostEqual(result.get("supportive"), 0.75)
        self.assertEqual(result.get("organization_name"), "Acme")

    def test_parse_llm_response_edge_cases(self):
        self.assertIsNone(parse_llm_response("nonsense"))

        bad_json = "```json\n{bad}\n```"
        self.assertIsNone(parse_llm_response(bad_json))

        text_yes = "```json\n{\"supportive\": \"yes\"}\n```"
        self.assertEqual(parse_llm_response(text_yes)["supportive"], 1.0)

        text_no = "```json\n{\"supportive\": \"no\"}\n```"
        self.assertEqual(parse_llm_response(text_no)["supportive"], 0.0)

        text_str_number = "```json\n{\"supportive\": \"0.25\"}\n```"
        self.assertAlmostEqual(parse_llm_response(text_str_number)["supportive"], 0.25)

        text_out_of_range = "```json\n{\"supportive\": 1.5}\n```"
        result = parse_llm_response(text_out_of_range)
        self.assertIsNotNone(result)
        self.assertIsNone(result.get("supportive"))

    def test_parse_llm_response_no_label(self):
        text = (
            "Intro.\n"
            "```\n"
            '{"supportive": 0.6}'
            "\n```"
        )
        result = parse_llm_response(text)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertAlmostEqual(result.get("supportive"), 0.6)

    def test_parse_llm_response_single_quotes(self):
        text = (
            "Intro.\n"
            "```json\n"
            "{'supportive': '0.3'}"
            "\n```"
        )
        result = parse_llm_response(text)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertAlmostEqual(result.get("supportive"), 0.3)

    @patch('company_lookup.openai.OpenAI')
    def test_cache_reused_for_same_seed(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = type(
            'Resp',
            (),
            {'choices': [type('Choice', (), {'message': {'content': 'ok'}})]},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pathlib.Path.home', return_value=pathlib.Path(tmpdir)):
                with patch.dict(os.environ, {
                    "OPENAI_API_KEY": "test-key",
                    "OPENAI_SEED": "123",
                }):
                    fetch_company_web_info("Acme Corp", model="test-model")
                    fetch_company_web_info("Acme Corp", model="test-model")

        self.assertEqual(mock_client.chat.completions.create.call_count, 1)

    @patch('company_lookup.openai.AsyncOpenAI', create=True)
    def test_async_cache_reused_for_same_seed(self, mock_async_openai):
        mock_client = mock_async_openai.return_value

        async def fake_create(**kwargs):
            return type(
                'Resp',
                (),
                {'choices': [type('Choice', (), {'message': {'content': 'ok'}})]},
            )

        mock_client.chat.completions.create = AsyncMock(side_effect=fake_create)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pathlib.Path.home', return_value=pathlib.Path(tmpdir)):
                with patch.dict(os.environ, {
                    'OPENAI_API_KEY': 'test-key',
                    'OPENAI_SEED': '123',
                }):
                    asyncio.run(async_fetch_company_web_info('Acme Corp', model='test-model'))
                    asyncio.run(async_fetch_company_web_info('Acme Corp', model='test-model'))

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
        report = generate_final_report(companies, stances)
        self.assertIn("Manufacturing: supportive company found", report)
        self.assertIn("Technology: no supportive company found", report)
        self.assertIn("Software: supportive company found", report)
        self.assertIn("Overall 2/3 companies are supportive", report)
        self.assertIn("Supportive companies by industry", report)
        self.assertIn("  Manufacturing: ########## (1/1)", report)
        self.assertIn("  Software: ########## (1/1)", report)
        self.assertIn("  Technology:  (0/1)", report)
        self.assertIn("Average stance per industry", report)
        self.assertIn("Support by IPO status", report)
        self.assertIn("Pre-IPO: 2/3", report)
        self.assertIn("Support by revenue", report)
        self.assertIn("Support by CB rank", report)
        self.assertIn("Supportive companies tend to be smaller based on employee counts.", report)
        self.assertIn("Supportive companies appear smaller based on combined size metric.", report)


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

if __name__ == '__main__':
    unittest.main()
