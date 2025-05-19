#!/usr/bin/env python3
"""CLI tool for fetching company info using the OpenAI-based lookup helper."""

from argparse import ArgumentParser
from pathlib import Path
import asyncio
from typing import List, Optional

from parser import read_companies_from_csv, Company
from company_lookup import async_fetch_company_web_info, parse_llm_response
import re


DEFAULT_MAX_LINES = 5
DEFAULT_MAX_CONCURRENCY = 5

# Mapping of lower-case industry aliases to their canonical names
_INDUSTRY_ALIASES = {
    "ai": "Artificial Intelligence",
    "artificial intelligence (ai)": "Artificial Intelligence",
}


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
    """Return a sanitized industry string for the company."""

    if not company.industries:
        return "Unknown"

    part = re.split(r"[;,]", company.industries)[0]
    part = part.strip()
    # Remove URLs and other obvious web references
    part = re.sub(r"https?://\S+|www\.[^\s]+", "", part)
    part = part.strip()
    # Strip trailing colons
    part = re.sub(r":+$", "", part).strip()
    # Remove common noise phrases such as "please visit" or "learn more"
    noise_patterns = [r"please visit.*", r"learn more.*"]
    for pattern in noise_patterns:
        part = re.split(pattern, part, flags=re.IGNORECASE)[0].strip()
    # Colon may appear before a removed phrase; strip again
    part = re.sub(r":+$", "", part).strip()

    # Discard unusually long text fragments that likely aren't simple
    # industry names.
    if len(part.split()) > 4:
        return "Unknown"

    if not part:
        return "Unknown"

    return _INDUSTRY_ALIASES.get(part.lower(), part)

def generate_final_report(companies: List[Company], stances: List[Optional[float]]) -> str:
    """Generate a more detailed summary of stance coverage per industry.

    ``stances`` should contain numeric values between 0 and 1 where higher
    numbers indicate stronger support for interoperability legislation.
    """

    from collections import defaultdict
    from statistics import mean

    try:
        from scipy.stats import ttest_ind
    except Exception:  # pragma: no cover - scipy optional
        ttest_ind = None

    industry_data = defaultdict(lambda: {"supportive": 0, "total": 0, "stances": []})
    support_emp: List[float] = []
    nonsupport_emp: List[float] = []

    for company, stance in zip(companies, stances):
        ind = _industry(company)
        info = industry_data[ind]
        info["total"] += 1
        if stance is not None:
            info["stances"].append(stance)

        emp = _employee_count(company)
        if stance is not None and stance >= 0.5:
            info["supportive"] += 1
            if emp is not None:
                support_emp.append(emp)
        else:
            if emp is not None:
                nonsupport_emp.append(emp)

    lines = ["Final Report:"]
    for ind in sorted(industry_data):
        if industry_data[ind]["supportive"] > 0:
            lines.append(f"- {ind}: supportive company found")
        else:
            lines.append(f"- {ind}: no supportive company found")

    total_support = sum(d["supportive"] for d in industry_data.values())
    total_companies = sum(d["total"] for d in industry_data.values())
    lines.append(f"Overall {total_support}/{total_companies} companies are supportive.")

    lines.append("\nSupportive companies by industry:")
    max_bar_width = 20
    total_support = sum(d["supportive"] for d in industry_data.values())
    if total_support == 0:
        total_support = 1
    for ind in sorted(industry_data):
        d = industry_data[ind]
        proportion = d["supportive"] / total_support
        bar_len = int(round(proportion * max_bar_width)) if d["supportive"] else 0
        # Ensure at least one character is shown for non-zero counts
        if d["supportive"] > 0 and bar_len == 0:
            bar_len = 1
        bar = "#" * bar_len
        lines.append(f"  {ind}: {bar} ({d['supportive']}/{d['total']})")

    lines.append("\nAverage stance per industry:")
    for ind in sorted(industry_data):
        st_list = industry_data[ind]["stances"]
        if st_list:
            lines.append(f"  {ind}: {mean(st_list):.2f}")
        else:
            lines.append(f"  {ind}: n/a")

    if support_emp and (support_emp or nonsupport_emp):
        avg_support_size = mean(support_emp)
        avg_total_size = mean(support_emp + nonsupport_emp)
        if avg_support_size < avg_total_size:
            lines.append("Supportive companies tend to be smaller based on employee counts.")
        else:
            lines.append("Supportive companies do not appear smaller based on employee counts.")

        if ttest_ind and len(support_emp) >= 2 and len(nonsupport_emp) >= 2:
            tstat, pval = ttest_ind(support_emp, nonsupport_emp, equal_var=False)
            lines.append(f"T-test comparing company size: t={tstat:.2f}, p={pval:.3f}")
        elif ttest_ind:
            lines.append("Not enough data for t-test of company size.")
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
