#!/usr/bin/env python3
"""CLI tool for fetching company info using the OpenAI-based lookup helper."""

from argparse import ArgumentParser
from pathlib import Path

from parser import read_companies_from_csv
from company_lookup import fetch_company_web_info


DEFAULT_MAX_LINES = 5


def main() -> None:
    parser = ArgumentParser(description="Fetch web summaries for companies listed in a CSV file")
    parser.add_argument("csv", type=Path, help="CSV file containing company data")
    parser.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES,
                        help=f"Number of lines to process (default: {DEFAULT_MAX_LINES})")
    args = parser.parse_args()

    companies = read_companies_from_csv(args.csv)
    total_rows = len(companies)

    if total_rows > 100:
        print(
            f"Warning: {args.csv} contains {total_rows} rows. "
            f"Only the first {args.max_lines} will be processed."
        )

    to_process = companies[: args.max_lines]

    for idx, company in enumerate(to_process, start=1):
        print(f"\n[{idx}/{len(to_process)}] Fetching info for: {company.organization_name}")
        try:
            summary = fetch_company_web_info(company.organization_name)
        except Exception as exc:
            print(f"Error fetching info for {company.organization_name}: {exc}")
            continue

        if summary:
            print(summary)
        else:
            print("No summary returned.")


if __name__ == "__main__":
    main()
