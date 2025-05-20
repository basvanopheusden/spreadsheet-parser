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


def _ipo_category(status: Optional[str]) -> str:
    """Return a coarse IPO category for a status string."""
    if not status:
        return "Unknown"
    status = status.lower()
    if "public" in status or "post" in status:
        return "Post-IPO"
    if "private" in status or "pre" in status:
        return "Pre-IPO"
    return "Unknown"


def _revenue_value(text: Optional[str]) -> Optional[float]:
    """Return an approximate revenue midpoint in millions."""
    if not text:
        return None
    text = text.replace(",", "").upper()
    matches = re.findall(r"\$?\s*(\d+(?:\.\d+)?)\s*([MB])", text)
    nums = []
    for num, unit in matches:
        try:
            val = float(num)
        except ValueError:
            continue
        if unit == "B":
            val *= 1000
        nums.append(val)
    if nums:
        if len(nums) == 1:
            return nums[0]
        return sum(nums) / len(nums)
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if m:
        val = float(m.group(1))
        if "B" in text:
            val *= 1000
        return val
    return None


def _revenue_category(text: Optional[str]) -> str:
    val = _revenue_value(text)
    if val is None:
        return "Unknown"
    if val < 10:
        return "<$10M"
    if val <= 100:
        return "$10M-$100M"
    return ">$100M"


def _cb_rank_value(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"\d+", text.replace(",", ""))
    return int(m.group()) if m else None


def _cb_rank_category(text: Optional[str]) -> str:
    val = _cb_rank_value(text)
    if val is None:
        return "Unknown"
    if val <= 1000:
        return "Top 1k"
    if val <= 5000:
        return "Top 5k"
    return "5k+"


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

def generate_final_report(
    companies: List[Company],
    stances: List[Optional[float]],
    subcategories: Optional[List[Optional[str]]] = None,
) -> str:
    """Generate a more detailed summary of stance coverage per industry.

    ``stances`` should contain numeric values between 0 and 1 where higher
    numbers indicate stronger support for interoperability legislation.
    """

    from collections import defaultdict, Counter
    from statistics import mean, median

    try:
        from scipy.stats import ttest_ind, chi2_contingency
    except Exception:  # pragma: no cover - scipy optional
        ttest_ind = None
        chi2_contingency = None

    industry_data = defaultdict(lambda: {"supportive": 0, "total": 0, "stances": []})
    subcat_data = defaultdict(lambda: {"supportive": 0, "total": 0})
    ipo_data = defaultdict(lambda: {"supportive": 0, "total": 0})
    revenue_data = defaultdict(lambda: {"supportive": 0, "total": 0})
    rank_data = defaultdict(lambda: {"supportive": 0, "total": 0})
    size_records = []  # (size_val, stance)
    emp_vals: List[float] = []
    rev_vals: List[float] = []
    rank_vals: List[int] = []
    support_emp: List[float] = []
    nonsupport_emp: List[float] = []

    sub_iter = subcategories if subcategories is not None else [None] * len(companies)
    for company, stance, subcat in zip(companies, stances, sub_iter):
        ind = _industry(company)
        info = industry_data[ind]
        info["total"] += 1
        if stance is not None:
            info["stances"].append(stance)

        sc = subcat or "Uncategorized"
        sc_info = subcat_data[sc]
        sc_info["total"] += 1
        if stance is not None and stance >= 0.5:
            sc_info["supportive"] += 1

        ipo_cat = _ipo_category(company.ipo_status)
        ipo_info = ipo_data[ipo_cat]
        ipo_info["total"] += 1
        if stance is not None and stance >= 0.5:
            ipo_info["supportive"] += 1

        rev_cat = _revenue_category(company.estimated_revenue_range)
        rev_info = revenue_data[rev_cat]
        rev_info["total"] += 1
        if stance is not None and stance >= 0.5:
            rev_info["supportive"] += 1

        rank_cat = _cb_rank_category(company.cb_rank)
        rank_info = rank_data[rank_cat]
        rank_info["total"] += 1
        if stance is not None and stance >= 0.5:
            rank_info["supportive"] += 1

        emp = _employee_count(company)
        if stance is not None and stance >= 0.5:
            info["supportive"] += 1
            if emp is not None:
                support_emp.append(emp)
        else:
            if emp is not None:
                nonsupport_emp.append(emp)

        if emp is not None:
            emp_vals.append(emp)
        rev_val = _revenue_value(company.estimated_revenue_range)
        if rev_val is not None:
            rev_vals.append(rev_val)
        rank_val = _cb_rank_value(company.cb_rank)
        if rank_val is not None:
            rank_vals.append(rank_val)

        size_records.append((emp, rev_val, rank_val, stance))
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

    lines.append("\nSupport by AI sub-category:")
    for cat in sorted(subcat_data):
        d = subcat_data[cat]
        lines.append(f"  {cat}: {d['supportive']}/{d['total']} supportive")

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

    lines.append("\nSupport by IPO status:")
    for cat in sorted(ipo_data):
        d = ipo_data[cat]
        lines.append(f"  {cat}: {d['supportive']}/{d['total']} supportive")
    if chi2_contingency and len(ipo_data) > 1:
        table = [
            [ipo_data[c]["supportive"] for c in sorted(ipo_data)],
            [ipo_data[c]["total"] - ipo_data[c]["supportive"] for c in sorted(ipo_data)],
        ]
        chi2, p, _, _ = chi2_contingency(table)
        lines.append(f"Chi-squared test for IPO status: p={p:.3f}")
    elif chi2_contingency:
        lines.append("Not enough categories for chi-squared test of IPO status.")

    lines.append("\nSupport by revenue range:")
    for cat in sorted(revenue_data):
        d = revenue_data[cat]
        lines.append(f"  {cat}: {d['supportive']}/{d['total']} supportive")
    if chi2_contingency and len(revenue_data) > 1:
        table = [
            [revenue_data[c]["supportive"] for c in sorted(revenue_data)],
            [revenue_data[c]["total"] - revenue_data[c]["supportive"] for c in sorted(revenue_data)],
        ]
        chi2, p, _, _ = chi2_contingency(table)
        lines.append(f"Chi-squared test for revenue range: p={p:.3f}")
    elif chi2_contingency:
        lines.append(
            "Not enough categories for chi-squared test of revenue range."
        )

    lines.append("\nSupport by CB rank:")
    for cat in sorted(rank_data):
        d = rank_data[cat]
        lines.append(f"  {cat}: {d['supportive']}/{d['total']} supportive")
    if chi2_contingency and len(rank_data) > 1:
        table = [
            [rank_data[c]["supportive"] for c in sorted(rank_data)],
            [rank_data[c]["total"] - rank_data[c]["supportive"] for c in sorted(rank_data)],
        ]
        chi2, p, _, _ = chi2_contingency(table)
        lines.append(f"Chi-squared test for CB rank: p={p:.3f}")
    elif chi2_contingency:
        lines.append("Not enough categories for chi-squared test of CB rank.")

    if size_records:
        size_vals_support = []
        size_vals_non = []
        emp_min, emp_max = (min(emp_vals), max(emp_vals)) if emp_vals else (None, None)
        rev_min, rev_max = (min(rev_vals), max(rev_vals)) if rev_vals else (None, None)
        rank_min, rank_max = (min(rank_vals), max(rank_vals)) if rank_vals else (None, None)

        def norm(val, lo, hi):
            if val is None or lo is None or hi is None or hi == lo:
                return None
            return (val - lo) / (hi - lo)

        for emp, rev, rank, stance in size_records:
            parts = []
            n = norm(emp, emp_min, emp_max)
            if n is not None:
                parts.append(n)
            n = norm(rev, rev_min, rev_max)
            if n is not None:
                parts.append(n)
            n = norm(rank, rank_min, rank_max)
            if n is not None:
                parts.append(1 - n)  # lower rank means larger
            if parts:
                size = sum(parts) / len(parts)
                if stance is not None and stance >= 0.5:
                    size_vals_support.append(size)
                else:
                    size_vals_non.append(size)

        if size_vals_support:
            lines.append("\nAverage company size metric (0=small, 1=large):")
            lines.append(f"  Supportive: {mean(size_vals_support):.2f}")
            if size_vals_non:
                lines.append(f"  Non-supportive: {mean(size_vals_non):.2f}")
            if ttest_ind and len(size_vals_support) >= 2 and len(size_vals_non) >= 2:
                tstat, pval = ttest_ind(size_vals_support, size_vals_non, equal_var=False)
                lines.append(f"T-test comparing size metric: t={tstat:.2f}, p={pval:.3f}")
            elif ttest_ind:
                lines.append("Not enough data for t-test of size metric.")

    industries_all = [_industry(c) for c in companies]
    ipo_statuses = [c.ipo_status or "Unknown" for c in companies]
    emp_values = [
        e for c in companies if (e := _employee_count(c)) is not None
    ]

    lines.append("\nInput data statistics:")
    if industries_all:
        ind_name, ind_count = Counter(industries_all).most_common(1)[0]
        lines.append(f"Most common industry: {ind_name} ({ind_count})")
    if ipo_statuses:
        ipo_name, ipo_count = Counter(ipo_statuses).most_common(1)[0]
        lines.append(f"Most common IPO status: {ipo_name} ({ipo_count})")
    if emp_values:
        lines.append(
            f"Employee counts (min/median/max): {int(min(emp_values))} / {int(median(emp_values))} / {int(max(emp_values))}"
        )

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
    subcats: List[Optional[str]] = []
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
            subcats.append(None)
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
                    justification = None
                    subcat = None
                else:
                    stance_val = parsed.get("supportive")
                    justification = parsed.get("justification")
                    subcat = parsed.get("sub_category")
                stances.append(stance_val)
                subcats.append(subcat)
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
                    subcat or "",
                    summary_text,
                    stance_label,
                    justification or summary_text,
                    rank_str,
                ])
            else:
                stances.append(None)
                subcats.append(None)
                table_rows.append([
                    company.organization_name,
                    _industry(company),
                    "",
                    "",
                    "Unknown",
                    "",
                    "",
                ])

        else:
            stances.append(None)
            subcats.append(None)
            table_rows.append([
                company.organization_name,
                _industry(company),
                "",
                "",
                "Unknown",
                "",
                "",
            ])

    report = generate_final_report(companies, stances, subcats)
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
                "AI Sub-Category",
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
