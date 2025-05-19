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


if __name__ == "__main__":
    unittest.main()

