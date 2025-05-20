import unittest

from parser import Company
from spreadsheet_parser.quality import find_duplicate_names, rows_with_missing_fields


class TestQualityChecks(unittest.TestCase):
    def make_company(self, name, url="http://example.com"):
        return Company(
            organization_name=name,
            organization_name_url=url,
            estimated_revenue_range=None,
            ipo_status=None,
            operating_status=None,
            acquisition_status=None,
            company_type=None,
            number_of_employees=None,
            full_description=None,
            industries=["Tech"],
            headquarters_location=None,
            description=None,
            cb_rank=None,
        )

    def test_detect_duplicates(self):
        comps = [
            self.make_company("Acme"),
            self.make_company("Globex"),
            self.make_company("Acme"),
        ]
        dups = find_duplicate_names(comps)
        self.assertEqual(dups, ["Acme"])

    def test_missing_critical_fields(self):
        c1 = self.make_company("Acme", url=None)
        c2 = self.make_company("Globex", url="http://globex.com")
        rows = rows_with_missing_fields([c1, c2], ["organization_name_url"])
        self.assertEqual(rows, [0])


if __name__ == "__main__":
    unittest.main()
