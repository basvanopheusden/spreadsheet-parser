#!/usr/bin/env python3
"""CLI tool for fetching company info using the OpenAI-based lookup helper."""

from argparse import ArgumentParser
from pathlib import Path
import asyncio

from parser import read_companies_from_csv
from company_lookup import async_fetch_company_web_info


DEFAULT_MAX_LINES = 5
DEFAULT_MAX_CONCURRENCY = 5


def main() -> None:
    parser = ArgumentParser(description="Fetch web summaries for companies listed in a CSV file")
    parser.add_argument("csv", type=Path, help="CSV file containing company data")
    parser.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES,
                        help=f"Number of lines to process (default: {DEFAULT_MAX_LINES})")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=f"Maximum concurrent API requests (default: {DEFAULT_MAX_CONCURRENCY})",
    )
    args = parser.parse_args()

    companies = read_companies_from_csv(args.csv)
    total_rows = len(companies)

    if total_rows > 100:
        print(
            f"Warning: {args.csv} contains {total_rows} rows. "
            f"Only the first {args.max_lines} will be processed."
        )

    to_process = companies[: args.max_lines]

    asyncio.run(_run_async(to_process, args.max_concurrency))


async def _run_async(companies, max_concurrency: int) -> None:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def fetch(company):
        async with semaphore:
            return await async_fetch_company_web_info(company.organization_name)

    tasks = [asyncio.create_task(fetch(c)) for c in companies]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for idx, result in enumerate(results, start=1):
        company = companies[idx - 1]
        print(f"\n[{idx}/{len(companies)}] Result for: {company.organization_name}")
        if isinstance(result, Exception):
            print(f"Error fetching info for {company.organization_name}: {result}")
        elif result:
            print(result)
        else:
            print("No summary returned.")


if __name__ == "__main__":
    main()
