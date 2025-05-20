import sys
import types
import unittest

if "openai" not in sys.modules:

    def make_client(*args, **kwargs):
        return types.SimpleNamespace(
            responses=types.SimpleNamespace(create=lambda **kwargs: None)
        )

    openai_stub = types.SimpleNamespace(OpenAI=make_client)
    sys.modules["openai"] = openai_stub

from parser import Company

from lookup_companies import _industry


class TestIndustryHelper(unittest.TestCase):
    def make_company(self, industries: str):
        return Company(
            organization_name="TestCo",
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

    def test_strip_whitespace(self):
        company = self.make_company("  Software  ")
        self.assertEqual(_industry(company), "Software")

    def test_remove_urls(self):
        company = self.make_company("https://example.com")
        self.assertEqual(_industry(company), "Unknown")

        company2 = self.make_company("Software https://example.com")
        self.assertEqual(_industry(company2), "Software")

    def test_discard_long_text(self):
        company = self.make_company("This industry description has many words")
        self.assertEqual(_industry(company), "Unknown")

    def test_strip_trailing_colons(self):
        company = self.make_company("Software:")
        self.assertEqual(_industry(company), "Software")

    def test_ignore_noise_phrases(self):
        company = self.make_company("Software: please visit our site")
        self.assertEqual(_industry(company), "Software")

        company2 = self.make_company("Please visit https://example.com")
        self.assertEqual(_industry(company2), "Unknown")

    def test_additional_noise_handling(self):
        company = self.make_company("California")
        self.assertEqual(_industry(company), "Unknown")

        company2 = self.make_company("such as GPT-3")
        self.assertEqual(_industry(company2), "Unknown")

        company3 = self.make_company("Location: AI")
        self.assertEqual(_industry(company3), "Artificial Intelligence")


if __name__ == "__main__":
    unittest.main()
