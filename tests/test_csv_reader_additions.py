import pathlib
import unittest

from parser import read_companies_from_csv, read_companies_from_csvs

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

    def test_read_multiple_with_filename_defaults(self):
        import tempfile
        from pathlib import Path

        csv_text = (
            "Organization Name,IPO Status,Operating Status,Estimated Revenue Range,Number of Employees\n"
            "Acme Corp,,,,\n"
            "Globex,Public,Active,$10M-$50M,51-100\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "FS_ActivePrivate_1Bto100B_51plus.csv"
            p1.write_text(csv_text, encoding="utf-8")
            p2 = Path(tmpdir) / "FS_ActivePrivate_1Bto100B_101plus.csv"
            p2.write_text(csv_text, encoding="utf-8")
            with self.assertLogs("spreadsheet_parser.csv_reader", level="WARNING") as cm:
                companies = read_companies_from_csvs([p1, p2])
        self.assertEqual(len(companies), 4)
        self.assertEqual(companies[0].ipo_status, "Private")
        self.assertEqual(companies[0].operating_status, "Active")
        self.assertEqual(companies[0].estimated_revenue_range, "$1B to $100B")
        self.assertEqual(companies[0].number_of_employees, "51+")
        self.assertEqual(len(cm.output), 5)

    def test_revenue_range_contained_no_warning(self):
        import tempfile
        from pathlib import Path

        csv_text = (
            "Organization Name,IPO Status,Operating Status,Estimated Revenue Range,Number of Employees\n"
            "Acme Corp,Private,Active,$1B to $10B,51-100\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "FS_ActivePrivate_1Bto100B_51plus.csv"
            p.write_text(csv_text, encoding="utf-8")
            with self.assertNoLogs("spreadsheet_parser.csv_reader", level="WARNING"):
                companies = read_companies_from_csvs([p])
        self.assertEqual(len(companies), 1)

if __name__ == "__main__":
    unittest.main()
