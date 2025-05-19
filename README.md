# Spreadsheet Parser

This repository provides a minimal Python utility for loading CSV files that describe organizations.
The goal is to convert each row of the spreadsheet into a structured dataclass for easy processing
and later analysis.

## Usage

1. Ensure you have Python 3.10 or newer installed.
2. Install the required dependencies using `pip`:

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
The tool now fetches results in parallel using asynchronous API calls. You can
control the level of concurrency with the `--max-concurrency` flag (default is 5).

```bash
python lookup_companies.py path/to/companies.csv --max-lines 5 --max-concurrency 10
```

This will fetch summaries and then display only the final report. The output
also notes how many responses were retrieved from the local cache. Use the
`--output-dir` option to specify where a CSV table (`company_analysis.csv`) and
the text report (`final_report.txt`) should be saved.

## Report Generation

The lookup script finishes by producing a **final report** summarizing stance
coverage for each industry. The report now includes additional breakdowns such
as support by IPO status, revenue range and CB rank, along with a simple company
size metric that combines these factors. Basic statistics such as the number of
supportive companies per industry, average stance values and an ASCII bar chart
are shown. The text report is written to ``final_report.txt`` alongside a CSV
table of company summaries. When the optional ``scipy`` package is installed,
a t-test is performed to compare employee counts of supportive vs. non-supportive
companies.

Example snippet:

```text
Final Report:
- Manufacturing: supportive company found
- Technology: no supportive company found
Overall 2/3 companies are supportive.

Supportive companies by industry:
  Manufacturing: # (1/1)
  Technology:  (0/1)

Average stance per industry:
  Manufacturing: 0.80
  Technology: 0.40

Support by IPO status:
  Pre-IPO: 2/3
Support by revenue:
  Unknown: 2/3
Support by CB rank:
  Unknown: 2/3
```

## Running Tests

The project uses Python's built-in ``unittest`` framework. To execute the entire
test suite, run the following command from the repository root:

```bash
python -m unittest discover -s tests -v
```

No additional dependencies or environment variables are required for testing.
The tests provide their own ``OPENAI_API_KEY`` and stub out the ``openai``
package when it is not installed.
