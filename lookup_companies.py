#!/usr/bin/env python3
"""CLI tool for fetching company info using the OpenAI-based lookup helper."""

from argparse import ArgumentParser
from pathlib import Path
import asyncio
from typing import List, Optional

from parser import read_companies_from_csv, Company
from company_lookup import async_fetch_company_web_info, parse_llm_response
import re


DEFAULT_MAX_LINES = 10
DEFAULT_MAX_CONCURRENCY = 5


def _employee_count(company: Company) -> Optional[float]:
    """Return an approximate employee count for a company."""
    text = company.number_of_employees or ""
    digits = [int(d) for d in re.findall(r"\d+", text)]
    if not digits:
        return None
    if len(digits) == 1:
        return float(digits[0])
    return sum(digits[:2]) / 2.0


def _industry(company: Company) -> str:
    """Return the first listed industry for the company."""
    if not company.industries:
        return "Unknown"
    parts = re.split(r"[;,]", company.industries)
    return parts[0].strip() or "Unknown"


def generate_final_report(companies: List[Company], stances: List[Optional[float]]) -> str:
    """Generate a short summary of stance coverage per industry.

    ``stances`` should contain numeric values between 0 and 1 where higher
    numbers indicate stronger support for interoperability legislation.
    """
    industry_map = {}
    total_emp = []
    support_emp = []

    for company, stance in zip(companies, stances):
        ind = _industry(company)
        industry_map.setdefault(ind, False)
        emp = _employee_count(company)
        if emp is not None:
            total_emp.append(emp)
        if stance is not None and stance >= 0.5:
            industry_map[ind] = True
            if emp is not None:
                support_emp.append(emp)

    lines = ["Final Report:"]
    for ind in sorted(industry_map):
        if industry_map[ind]:
            lines.append(f"- {ind}: supportive company found")
        else:
            lines.append(f"- {ind}: no supportive company found")

    if support_emp and total_emp:
        avg_support = sum(support_emp) / len(support_emp)
        avg_total = sum(total_emp) / len(total_emp)
        if avg_support < avg_total:
            lines.append("Supportive companies tend to be smaller based on employee counts.")
        else:
            lines.append("Supportive companies do not appear smaller based on employee counts.")
    else:
        lines.append("Insufficient data to compare company sizes.")

    return "\n".join(lines)


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

    stances: List[Optional[float]] = []

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
            stances.append(None)
        elif result:
            print(result)
            stances.append(parse_llm_response(result))
        else:
            print("No summary returned.")
            stances.append(None)

    report = generate_final_report(companies, stances)
    print("\n" + report)


if __name__ == "__main__":
    main()
