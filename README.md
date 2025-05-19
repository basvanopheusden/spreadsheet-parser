# Spreadsheet Parser

This repository provides a minimal Python utility for loading CSV files that describe organizations.
The goal is to convert each row of the spreadsheet into a structured dataclass for easy processing
and later analysis.

## Usage

1. Ensure you have Python 3.10 or newer installed.
2. Prepare a CSV file with a header containing the following columns:
   - Organization Name
   - Organization Name URL
   - Estimated Revenue Range
   - IPO Status
   - Operating Status
   - Acquisition Status
   - Company Type
   - Number of Employees
   - Full Description
   - Industries
   - Headquarters Location
   - Description
   - CB Rank (Company)
3. Import the reader function and parse your data:

```python
from parser import read_companies_from_csv

companies = read_companies_from_csv("companies.csv")
for company in companies:
    print(company.organization_name, company.ipo_status)
```

The `Company` dataclass defined in `parser.py` mirrors the columns listed above and
serves as a convenient container for the parsed information.

This setup can be extended to support more advanced analysis or integration with
other tools that require structured data.
