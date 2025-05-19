import os
import sys
import types
import unittest
import tempfile
import pathlib
from unittest.mock import patch

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
from company_lookup import fetch_company_web_info, parse_llm_response


class TestFetchCompanyWebInfo(unittest.TestCase):
    @patch('company_lookup.openai.OpenAI')
    def test_prompt_includes_csv_details(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = type(
            'Resp',
            (),
            {'choices': [type('Choice', (), {'message': {'content': 'ok'}})]},
        )

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
            self.assertIn("Estimated Revenue Range: $10M-$50M", user_content)
            self.assertIn("Headquarters Location: New York, NY", user_content)
            self.assertIn("key 'stance'", user_content)

    def test_parse_llm_response(self):
        text = (
            "Acme summary.\n"
            "```json\n"
            '{"stance": "supportive"}'
            "\n```"
        )
        self.assertEqual(parse_llm_response(text), "supportive")

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


if __name__ == '__main__':
    unittest.main()
