import pathlib
import unittest
from parser import read_companies_from_csv


class TestReadCompaniesFromCSV(unittest.TestCase):
    def test_read(self):
        path = pathlib.Path(__file__).parent / "data" / "companies.csv"
        companies = read_companies_from_csv(path)

        self.assertEqual(len(companies), 3)

        first = companies[0]
        self.assertEqual(first.organization_name, "Acme Corp")
        self.assertEqual(first.cb_rank, "1")

        names = [c.organization_name for c in companies]
        self.assertEqual(names, ["Acme Corp", "Globex Inc", "Initech"])

    def test_semicolon_delimiter(self):
        path = pathlib.Path(__file__).parent / "data" / "companies_semicolon.csv"
        companies = read_companies_from_csv(path)
        self.assertEqual(len(companies), 3)
        self.assertEqual(companies[1].organization_name, "Globex Inc")

    def test_header_variations(self):
        path = pathlib.Path(__file__).parent / "data" / "companies_header_variation.csv"
        companies = read_companies_from_csv(path)
        self.assertEqual(len(companies), 3)
        self.assertEqual(
            companies[2].organization_name_url, "https://initech.example.com"
        )

    def test_missing_columns(self):
        path = pathlib.Path(__file__).parent / "data" / "companies_missing_columns.csv"
        companies = read_companies_from_csv(path)
        self.assertEqual(len(companies), 3)
        self.assertIsNone(companies[0].full_description)

    def test_extra_column_ignored(self):
        path = (
            pathlib.Path(__file__).parent
            / "data"
            / "companies_malformed_extra_column.csv"
        )
        companies = read_companies_from_csv(path)
        self.assertEqual(len(companies), 2)
        self.assertEqual(companies[0].organization_name, "Acme Corp")

    def test_unbalanced_quote(self):
        path = pathlib.Path(__file__).parent / "data" / "companies_malformed_quote.csv"
        companies = read_companies_from_csv(path)
        self.assertEqual(len(companies), 1)
        self.assertTrue(companies[0].organization_name.startswith("Broken Co"))

    def test_misencoded_text(self):
        path = (
            pathlib.Path(__file__).parent / "data" / "companies_malformed_encoding.csv"
        )
        companies = read_companies_from_csv(path)
        self.assertEqual(len(companies), 1)
        self.assertIn("√", companies[0].organization_name)


if __name__ == "__main__":
    unittest.main()
