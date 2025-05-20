import pathlib
import unittest
from parser import read_companies_from_csv, read_companies_from_xlsx
from spreadsheet_parser.models import CANONICAL_HEADERS

try:
    import openpyxl  # type: ignore
    HAS_OPENPYXL = True
except Exception:  # pragma: no cover - optional dependency
    HAS_OPENPYXL = False


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

    def test_excludes_non_business(self):
        path = pathlib.Path(__file__).parent / "data" / "companies_non_business.csv"
        companies = read_companies_from_csv(path)
        names = [c.organization_name for c in companies]
        self.assertEqual(names, ["Acme Corp", "Globex Inc"])


@unittest.skipUnless(HAS_OPENPYXL, "openpyxl not installed")
class TestReadCompaniesFromXLSX(unittest.TestCase):
    def setUp(self):
        import tempfile
        import openpyxl  # type: ignore

        self.tempfile = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(list(CANONICAL_HEADERS.values()))
        ws.append(
            [
                "Acme Corp",
                "https://acme.example.com",
                "$10M-$50M",
                "Private",
                "Operating",
                None,
                "For Profit",
                "100-250",
                "Acme Corp specializes in gadgets and gizmos.",
                "Manufacturing",
                "New York, NY",
                "Leading gadget manufacturer",
                "1",
            ]
        )
        ws.append(
            [
                "Globex Inc",
                "https://globex.example.com",
                "$50M-$100M",
                "Public",
                "Operating",
                None,
                "For Profit",
                "500-1000",
                "Globex provides a variety of high-tech solutions.",
                "Technology",
                "San Francisco, CA",
                "Innovative tech company",
                "2",
            ]
        )
        ws.append(
            [
                "Initech",
                "https://initech.example.com",
                "<$10M",
                "Private",
                "Operating",
                None,
                "For Profit",
                "50-100",
                "Initech focuses on enterprise software.",
                "Software",
                "Austin, TX",
                "Enterprise software vendor",
                "3",
            ]
        )
        wb.save(self.tempfile.name)

    def tearDown(self):
        import os

        os.unlink(self.tempfile.name)

    def test_read_xlsx(self):
        companies = read_companies_from_xlsx(self.tempfile.name)
        self.assertEqual(len(companies), 3)
        self.assertEqual(companies[0].organization_name, "Acme Corp")


if __name__ == "__main__":
    unittest.main()
