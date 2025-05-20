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
4. Import the reader function and parse your data. Both CSV and XLSX files are
   supported:

```python
from spreadsheet_parser import read_companies_from_csv, read_companies_from_xlsx

companies = read_companies_from_csv("companies.csv")
# or
companies = read_companies_from_xlsx("companies.xlsx")
for company in companies:
    print(company.organization_name, company.ipo_status)
```

The `Company` dataclass defined in `spreadsheet_parser.models` mirrors the columns listed above and
serves as a convenient container for the parsed information.

This setup can be extended to support more advanced analysis or integration with
other tools that require structured data.

## Optional Company Lookup Stub

The repository also contains a simple helper, `fetch_company_web_info`, in
`company_lookup.py`. It sends a prompt to OpenAI's API to retrieve a summary of a
company from the web. The helper expects an ``OPENAI_API_KEY`` environment
variable and, optionally, an ``OPENAI_MODEL`` variable to choose the model. If no
model is specified, it defaults to ``gpt-4o``. The command-line tool described
below also accepts a ``--model-name`` argument to override this value.

The project now relies on ``openai==1.79.0``. Direct API calls use the updated
``OpenAI`` client:

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-4.1",
    tools=[{"type": "web_search_preview"}],
    input="What was a positive news story from today?",
)
print(response.output_text)
```

You may provide just the company name or pass a `Company` object returned from
`read_companies_from_csv` or `read_companies_from_xlsx`. 
When a dataclass is supplied, all details from the CSV
are included in the prompt sent to the LLM so that it can give a more informed
summary. Responses are parsed with `parse_llm_response`, which now checks that
the JSON payload contains `supportive` and `is_business` keys. If either key is
missing the function returns `None` (or raises when called with
`raise_on_missing=True`). In all cases the automated results should be reviewed
manually.
## Lookup Script

For convenience, the repository provides a small CLI script, `lookup_companies.py`,
which reads a CSV file and uses `fetch_company_web_info` to retrieve summaries for
each company. The script processes only a limited number of rows (default is 5)
and issues a warning if the CSV contains more than 100 lines.
The tool now fetches results in parallel using asynchronous API calls. You can
control the level of concurrency with the `--max-concurrency` flag (default is 5)
and select the OpenAI model with `--model-name` (default is `gpt-4o`).

```bash
python lookup_companies.py path/to/companies.csv \
    --max-lines 5 --max-concurrency 10 --model-name gpt-4o
```

This will fetch summaries and then display only the final report. The output
also notes how many responses were retrieved from the local cache. Use the
`--output-dir` option to specify where a CSV table (`company_analysis.csv`) and
the text report (`final_report.txt`) should be saved.

## Report Generation

The lookup script finishes by producing a **final report** summarizing stance
coverage for each industry. Parsed summaries are validated so that the
`supportive` stance score and `is_business` flag are always present, but the
results should still be reviewed manually. The report includes basic statistics such as the
number of supportive companies per industry, average percentile scores and a simple
ASCII bar chart. It also lists common categories found in the input data and
provides a short summary of numeric value distributions. Support levels are
broken down by IPO status, revenue range and CB rank, and the output includes a
simple company size metric that combines employee counts with these fields. The
report lists the top three most supportive companies for each AI sub-category.
Stance scores in the report are expressed as percentile ranks so they always
fall between 0 and 1. All metrics and statistical tests use these percentile
values rather than the raw scores. The text report is written to
``final_report.txt`` alongside a CSV table of company summaries. When the
optional ``scipy`` package is installed, statistical tests are included in the
report. T-tests compare employee counts and the derived size metric between
supportive and non-supportive companies, while chi-squared tests examine
whether IPO status, revenue range or CB rank are associated with support
levels.
The full text report is then submitted to the same language model and rewritten
as a concise scientific paper abstract. This abstract is saved as
``abstract.txt`` in the chosen output directory.

If the optional ``matplotlib`` package is installed, a ``support_by_subcat.png``
image is also created showing supportive and opposing company counts per AI
sub-category. The bars include simple binomial error bars and are colored
green/red for support and opposition respectively.

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
  Manufacturing: 0.50
  Technology: 0.17
Input data statistics:
Most common industry: Manufacturing (1)
Most common IPO status: Unknown (3)
Employee counts (min/median/max): 75 / 175 / 750
Support by IPO status:
  Unknown: 2/3 supportive
Support by revenue range:
  Unknown: 2/3 supportive
Support by CB rank:
  Unknown: 2/3 supportive
Average company size metric (0=small, 1=large):
  Supportive: 0.07
  Non-supportive: 1.00
T-test comparing company size: t=0.50, p=0.650
T-test comparing size metric: t=0.45, p=0.700
Chi-squared test for IPO status: p=0.500
Chi-squared test for revenue range: p=0.500
Chi-squared test for CB rank: p=0.500
```

## Percentile Ranking

The helper function ``percentile_ranks`` can be used to normalize stance values.
It converts a list of numeric scores to percentile ranks between 0 and 1. Any
``None`` or non-numeric values result in ``None`` in the returned list. The
final report automatically applies this transformation so all stance statistics
use percentile scores.

```python
from spreadsheet_parser import percentile_ranks

scores = [0.9, 0.5, 0.7]
print(percentile_ranks(scores))
# [0.83, 0.17, 0.50]
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

## License

This project is licensed under the [MIT License](LICENSE).
