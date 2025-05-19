# Spreadsheet Parser

This repository provides a minimal Python utility for loading CSV files that describe organizations.
The goal is to convert each row of the spreadsheet into a structured dataclass for easy processing
and later analysis.

## Usage

1. Ensure you have Python 3.10 or newer installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare a CSV file with a header containing the following columns:
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
4. Import the reader function and parse your data:

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

## Optional Company Lookup Stub

The repository also contains a simple helper, `fetch_company_web_info`, in
`company_lookup.py`. It sends a prompt to OpenAI's API to retrieve a summary of a
company from the web. The helper expects an ``OPENAI_API_KEY`` environment
variable and, optionally, an ``OPENAI_MODEL`` variable to choose the model. If no
model is specified, it defaults to ``gpt-4o``.

You may provide just the company name or pass a `Company` object returned from
`read_companies_from_csv`. When a dataclass is supplied, all details from the CSV
are included in the prompt sent to the LLM so that it can give a more informed
summary.
## Lookup Script

For convenience, the repository provides a small CLI script, `lookup_companies.py`,
which reads a CSV file and uses `fetch_company_web_info` to retrieve summaries for
each company. The script processes only a limited number of rows (default is 5)
and issues a warning if the CSV contains more than 100 lines.

```bash
python lookup_companies.py path/to/companies.csv --max-lines 5
```

This will print summaries for the first few companies in the spreadsheet.
