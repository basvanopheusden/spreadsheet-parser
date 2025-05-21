import unittest

from parser import Company
from spreadsheet_parser.quality import (
    find_duplicate_names,
    rows_with_missing_fields,
    revenue_employee_outliers,
    rows_with_garbled_text,
)


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

    def test_revenue_employee_outliers(self):
        c1 = self.make_company("TinyCo")
        c1.number_of_employees = "1-10"
        c1.estimated_revenue_range = "$1B to $10B"
        c2 = self.make_company("BigCo")
        c2.number_of_employees = "500-1000"
        c2.estimated_revenue_range = "$1B to $10B"
        rows = revenue_employee_outliers([c1, c2])
        self.assertEqual(rows, [0])

    def test_rows_with_garbled_text(self):
        normal = self.make_company("NormalCo")
        garbled = self.make_company("WeirdCo")
        garbled.industries = ["http://example.com industry"]
        garbled.headquarters_location = "City: http://bad"
        rows = rows_with_garbled_text([normal, garbled])
        self.assertEqual(rows, [1])


if __name__ == "__main__":
    unittest.main()
