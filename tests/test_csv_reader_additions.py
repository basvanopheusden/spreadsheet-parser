import types
import sys
import pathlib
import unittest

if "openai" not in sys.modules:
    openai_stub = types.SimpleNamespace(ChatCompletion=types.SimpleNamespace(create=lambda **kwargs: None), OpenAI=lambda *a, **kw: None)
    sys.modules["openai"] = openai_stub

from parser import read_companies_from_csv

class TestCSVReaderAdditions(unittest.TestCase):
    def test_employee_range_correction(self):
        path = pathlib.Path(__file__).parent / "data" / "company_sample.csv"
        companies = read_companies_from_csv(path)
        self.assertEqual(companies[1].number_of_employees, "11-50")
        self.assertEqual(companies[2].number_of_employees, "1-10")

    def test_fallback_decoding(self):
        path = pathlib.Path(__file__).parent / "data" / "company_sample_bad_encoding.csv"
        companies = read_companies_from_csv(path)
        self.assertEqual(len(companies), 3)
        self.assertTrue(companies[0].organization_name.startswith("OpenAI"))

if __name__ == "__main__":
    unittest.main()
