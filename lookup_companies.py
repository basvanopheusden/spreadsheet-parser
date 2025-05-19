#!/usr/bin/env python3
"""CLI tool for fetching company info using the OpenAI-based lookup helper."""

from argparse import ArgumentParser
from pathlib import Path
import asyncio
import csv
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

    parts = re.split(r"[;,]", company.industries)

    noise_patterns = [
        r"please visit.*",
        r"learn more.*",
        r"such as.*",
        r"for more.*",
        r"based in.*",
    ]

    skip_tokens = {
        "cost",
        "city",
        "cities",
        "entity",
        "entities",
        "next-gen",
        "proprietary process",
        "california",
    }

    for raw in parts:
        part = raw.strip()
        # Remove URLs and other obvious web references
        part = re.sub(r"https?://\S+|www\.[^\s]+", "", part).strip()
        part = re.sub(r":+$", "", part).strip()
        for pattern in noise_patterns:
            part = re.split(pattern, part, flags=re.IGNORECASE)[0].strip()
        part = re.sub(r":+$", "", part).strip()

        if ":" in part:
            maybe = part.split(":", 1)[1].strip()
            if maybe and len(maybe.split()) <= 4:
                part = maybe

        if not part:
            continue
        if len(part.split()) > 4:
            continue
        if re.search(r"\d", part):
            continue
        if part.lower() in skip_tokens or re.search(
            r"\b(?:city|cities|county|state|province|cost|entities?|next-gen|gpt-\d+|proprietary process)\b",
            part,
            re.IGNORECASE,
        ):
            continue

        return _INDUSTRY_ALIASES.get(part.lower(), part)

    return "Unknown"


def _cb_rank_value(company: Company) -> Optional[float]:
    """Return the Crunchbase rank as a float if available."""
    text = (company.cb_rank or "").replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def _revenue_value(company: Company) -> Optional[float]:
    """Parse the estimated revenue range and return an average in millions."""
    text = (company.estimated_revenue_range or "").replace(",", "").upper()
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if not nums:
        return None
    values = [float(n) for n in nums]
    avg = sum(values) / len(values)
    if "B" in text or "BILLION" in text:
        avg *= 1000
    return avg


def _company_size(company: Company) -> Optional[float]:
    """Compute a simple size metric using employees, revenue and CB rank."""
    emp = _employee_count(company)
    rev = _revenue_value(company)
    rank = _cb_rank_value(company)

    scores = []
    if emp is not None:
        scores.append(emp)
    if rev is not None:
        scores.append(rev)
    if rank is not None:
        scores.append(max(0.0, 10000 - rank))
    if not scores:
        return None
    return sum(scores) / len(scores)

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
    ipo_data = defaultdict(lambda: {"supportive": 0, "total": 0})
    revenue_data = defaultdict(lambda: {"supportive": 0, "total": 0})
    rank_data = defaultdict(lambda: {"supportive": 0, "total": 0})
    support_emp: List[float] = []
    nonsupport_emp: List[float] = []
    support_size: List[float] = []
    nonsupport_size: List[float] = []

    for company, stance in zip(companies, stances):
        ind = _industry(company)
        info = industry_data[ind]
        info["total"] += 1
        if stance is not None:
            info["stances"].append(stance)

        emp = _employee_count(company)
        size_score = _company_size(company)
        is_supportive = stance is not None and stance >= 0.5

        if is_supportive:
            info["supportive"] += 1
            if emp is not None:
                support_emp.append(emp)
            if size_score is not None:
                support_size.append(size_score)
        else:
            if emp is not None:
                nonsupport_emp.append(emp)
            if size_score is not None:
                nonsupport_size.append(size_score)

        ipo_cat = "Post-IPO" if (company.ipo_status or "").strip().lower() == "public" else "Pre-IPO"
        ipo_data[ipo_cat]["total"] += 1
        if is_supportive:
            ipo_data[ipo_cat]["supportive"] += 1

        rev_val = _revenue_value(company)
        if rev_val is None:
            rev_cat = "Unknown"
        elif rev_val >= 50:
            rev_cat = "Over $50M"
        else:
            rev_cat = "Under $50M"
        revenue_data[rev_cat]["total"] += 1
        if is_supportive:
            revenue_data[rev_cat]["supportive"] += 1

        rank_val = _cb_rank_value(company)
        if rank_val is None:
            rank_cat = "Unknown"
        elif rank_val <= 5000:
            rank_cat = "Top 5k"
        else:
            rank_cat = "5k+"
        rank_data[rank_cat]["total"] += 1
        if is_supportive:
            rank_data[rank_cat]["supportive"] += 1

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

    lines.append("\nSupport by IPO status:")
    for cat in sorted(ipo_data):
        d = ipo_data[cat]
        lines.append(f"  {cat}: {d['supportive']}/{d['total']}")

    lines.append("\nSupport by revenue:")
    for cat in sorted(revenue_data):
        d = revenue_data[cat]
        lines.append(f"  {cat}: {d['supportive']}/{d['total']}")

    lines.append("\nSupport by CB rank:")
    for cat in sorted(rank_data):
        d = rank_data[cat]
        lines.append(f"  {cat}: {d['supportive']}/{d['total']}")

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

    if support_size and (support_size or nonsupport_size):
        avg_support_metric = mean(support_size)
        avg_total_metric = mean(support_size + nonsupport_size)
        if avg_support_metric < avg_total_metric:
            lines.append(
                "Supportive companies appear smaller based on combined size metric."
            )
        else:
            lines.append(
                "Supportive companies do not appear smaller based on combined size metric."
            )
    else:
        lines.append("Insufficient data to compare combined size metric.")

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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where the final report and table will be saved",
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

    asyncio.run(_run_async(to_process, args.max_concurrency, args.output_dir))


async def _run_async(companies, max_concurrency: int, output_dir: Path) -> None:
    semaphore = asyncio.Semaphore(max_concurrency)

    stances: List[Optional[float]] = []
    cached_count = 0
    table_rows: List[List[str]] = []

    async def fetch(company):
        async with semaphore:
            return await async_fetch_company_web_info(
                company.organization_name,
                return_cache_info=True,
            )

    tasks = [asyncio.create_task(fetch(c)) for c in companies]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for company, result in zip(companies, results):
        if isinstance(result, Exception):
            stances.append(None)
            table_rows.append([
                company.organization_name,
                _industry(company),
                "",
                "Unknown",
                "",
                "",
            ])
            continue
        elif result:
            content, cached = result
            if cached:
                cached_count += 1
            if content:
                print(content)
                parsed = parse_llm_response(content)
                if parsed is None:
                    stance_val = None
                else:
                    stance_val = parsed.get("supportive")
                stances.append(stance_val)
                summary_text = re.split(r"```(?:json)?\s*\{.*?\}\s*```", content, flags=re.DOTALL)[0].strip()
                if stance_val is None:
                    stance_label = "Unknown"
                    rank_str = ""
                else:
                    stance_label = "Support" if stance_val >= 0.5 else "Oppose"
                    rank_str = f"{stance_val:.2f}"
                table_rows.append([
                    company.organization_name,
                    _industry(company),
                    summary_text,
                    stance_label,
                    summary_text,
                    rank_str,
                ])
            else:
                stances.append(None)
                table_rows.append([
                    company.organization_name,
                    _industry(company),
                    "",
                    "Unknown",
                    "",
                    "",
                ])

        else:
            stances.append(None)
            table_rows.append([
                company.organization_name,
                _industry(company),
                "",
                "Unknown",
                "",
                "",
            ])

    report = generate_final_report(companies, stances)
    print(report)
    print(f"Cached responses used: {cached_count}")

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "company_analysis.csv"
    with table_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Company Name",
                "Industry",
                "Business Model Summary",
                "Likely Stance on Interoperability",
                "Qualitative Justification",
                "Quantitative Ranking",
            ]
        )
        for row in table_rows:
            writer.writerow(row)

    report_path = output_dir / "final_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"Output table saved to {table_path}")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
