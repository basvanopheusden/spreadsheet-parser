import asyncio
import csv
import os
import re
import openai
import inspect
from .models import Company
from pathlib import Path
from typing import List, Optional, Tuple

from .llm import (
    parse_llm_response,
    async_report_to_abstract,
)
from datetime import date, datetime

DEFAULT_MAX_LINES = 5
DEFAULT_MAX_CONCURRENCY = 5

# Mapping of lower-case industry aliases to their canonical names
_INDUSTRY_ALIASES = {
    "ai": "Artificial Intelligence",
    "artificial intelligence (ai)": "Artificial Intelligence",
}


def _employee_count(company: Company) -> Optional[float]:
    """Return an approximate employee count for a company."""
    text = company.number_of_employees
    if text is None:
        return None

    if isinstance(text, (int, float)):
        return float(text)

    if isinstance(text, (datetime, date)):
        return None

    text = str(text)
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
    """Return the Crunchbase rank as an integer if possible."""

    if text is None:
        return None

    # ``cb_rank`` may come in as an int or float when read from XLSX files.
    # Convert it to ``str`` so that the regex search works consistently.
    text_str = str(text)

    m = re.search(r"\d+", text_str.replace(",", ""))
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


def _industry_list(company: Company) -> List[str]:
    """Return a list of sanitized industries for the company."""

    if not company.industries:
        return ["Unknown"]

    if isinstance(company.industries, str):
        parts = re.split(r"[;,]", company.industries)
    else:
        parts = company.industries

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

    results = []
    for raw in parts:
        part = str(raw).strip()
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

        results.append(_INDUSTRY_ALIASES.get(part.lower(), part))

    return results or ["Unknown"]


def _industry(company: Company) -> str:
    """Return the first sanitized industry string for the company."""

    return _industry_list(company)[0]


from .csv_reader import _is_business


def percentile_ranks(values: List[Optional[float]]) -> List[Optional[float]]:
    """Return percentile ranks between 0 and 1 for each numeric value.

    ``None`` or non-numeric entries yield ``None`` in the output list. Tied
    values receive the same percentile based on their average rank.
    """

    pairs = [(idx, float(v)) for idx, v in enumerate(values) if isinstance(v, (int, float))]
    if not pairs:
        return [None] * len(values)

    pairs.sort(key=lambda x: x[1])
    n = len(pairs)
    result: List[Optional[float]] = [None] * len(values)
    i = 0
    while i < n:
        val = pairs[i][1]
        j = i + 1
        while j < n and pairs[j][1] == val:
            j += 1
        rank_avg = (i + 1 + j) / 2.0
        perc = (rank_avg - 0.5) / n
        for k in range(i, j):
            result[pairs[k][0]] = perc
        i = j

    return result


def generate_final_report(
    companies: List[Company],
    stances: List[Optional[float]],
    subcategories: Optional[List[Optional[str]]] = None,
    justifications: Optional[List[Optional[str]]] = None,
    is_business_flags: Optional[List[Optional[bool]]] = None,
) -> str:
    """Generate a more detailed summary of stance coverage per industry.

    ``stances`` should contain numeric values between 0 and 1 where higher
    numbers indicate stronger support for interoperability legislation. The
    scores are converted to percentile ranks in the ``[0, 1]`` range and all
    subsequent metrics, including statistical tests, are computed from these
    percentile values.
    ``justifications`` may provide short explanations for each company's
    stance. When supplied, a few example lines with these justifications are
    included in the final report.
    ``is_business_flags`` allows callers to specify whether each row represents
    a for-profit business. Any entries marked as ``False`` are ignored, as are
    company names that fail the internal ``_is_business`` heuristic.
    """

    from collections import Counter, defaultdict
    from statistics import mean, median

    try:
        from scipy.stats import chi2_contingency, ttest_ind
    except Exception:  # pragma: no cover - scipy optional
        ttest_ind = None
        chi2_contingency = None

    emp_ttest: Optional[Tuple[float, float]] = None
    size_ttest: Optional[Tuple[float, float]] = None
    ipo_chi2: Optional[float] = None
    rev_chi2: Optional[float] = None
    rank_chi2: Optional[float] = None

    industry_data = defaultdict(lambda: {"supportive": 0, "total": 0, "stances": []})
    subcat_data = defaultdict(lambda: {"supportive": 0, "total": 0})
    subcat_top = defaultdict(list)
    ipo_data = defaultdict(lambda: {"supportive": 0, "total": 0})
    revenue_data = defaultdict(lambda: {"supportive": 0, "total": 0})
    rank_data = defaultdict(lambda: {"supportive": 0, "total": 0})
    size_records = []  # (size_val, stance)
    emp_vals: List[float] = []
    rev_vals: List[float] = []
    rank_vals: List[int] = []
    support_emp: List[float] = []
    nonsupport_emp: List[float] = []

    percentiles = percentile_ranks(stances)
    total_companies = 0
    total_support = 0
    # Use percentile ranks for all downstream calculations
    sub_iter = subcategories if subcategories is not None else [None] * len(companies)
    biz_iter = is_business_flags if is_business_flags is not None else [None] * len(companies)
    for company, stance, perc, subcat, is_biz in zip(
        companies,
        stances,
        percentiles,
        sub_iter,
        biz_iter,
    ):
        if is_biz is False or not _is_business(company.organization_name):
            continue
        total_companies += 1
        if perc is not None and perc >= 0.5:
            total_support += 1

        for ind in _industry_list(company):
            info = industry_data[ind]
            info["total"] += 1
            if perc is not None:
                info["stances"].append(perc)
                if perc >= 0.5:
                    info["supportive"] += 1

        sc = subcat or "Uncategorized"
        sc_info = subcat_data[sc]
        sc_info["total"] += 1
        if perc is not None:
            subcat_top[sc].append((perc, company.organization_name))
            if perc >= 0.5:
                sc_info["supportive"] += 1

        ipo_cat = _ipo_category(company.ipo_status)
        ipo_info = ipo_data[ipo_cat]
        ipo_info["total"] += 1
        if perc is not None and perc >= 0.5:
            ipo_info["supportive"] += 1

        rev_cat = _revenue_category(company.estimated_revenue_range)
        rev_info = revenue_data[rev_cat]
        rev_info["total"] += 1
        if perc is not None and perc >= 0.5:
            rev_info["supportive"] += 1

        rank_cat = _cb_rank_category(company.cb_rank)
        rank_info = rank_data[rank_cat]
        rank_info["total"] += 1
        if perc is not None and perc >= 0.5:
            rank_info["supportive"] += 1

        emp = _employee_count(company)
        if perc is not None and perc >= 0.5:
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

        size_records.append((emp, rev_val, rank_val, perc))
    lines = ["Final Report:"]
    for ind in sorted(industry_data):
        if industry_data[ind]["supportive"] > 0:
            lines.append(f"- {ind}: supportive company found")
        else:
            lines.append(f"- {ind}: no supportive company found")

    lines.append(f"Overall {total_support}/{total_companies} companies are supportive.")

    lines.append("\nSupportive companies by industry:")
    max_bar_width = 20
    industry_support_total = sum(d["supportive"] for d in industry_data.values())
    if industry_support_total == 0:
        industry_support_total = 1
    for ind in sorted(industry_data):
        d = industry_data[ind]
        proportion = d["supportive"] / industry_support_total
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
        top = sorted(
            (pair for pair in subcat_top.get(cat, []) if pair[0] is not None),
            key=lambda x: x[0],
            reverse=True,
        )[:3]
        if top:
            names = ", ".join(name for _, name in top)
            lines.append(f"    Top companies: {names}")

    if support_emp and (support_emp or nonsupport_emp):
        avg_support_size = mean(support_emp)
        avg_total_size = mean(support_emp + nonsupport_emp)
        if avg_support_size < avg_total_size:
            lines.append(
                "Supportive companies tend to be smaller based on employee counts."
            )
        else:
            lines.append(
                "Supportive companies do not appear smaller based on employee counts."
            )

        if ttest_ind and len(support_emp) >= 2 and len(nonsupport_emp) >= 2:
            tstat, pval = ttest_ind(support_emp, nonsupport_emp, equal_var=False)
            emp_ttest = (tstat, pval)
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
            [
                ipo_data[c]["total"] - ipo_data[c]["supportive"]
                for c in sorted(ipo_data)
            ],
        ]
        chi2, p, _, _ = chi2_contingency(table)
        ipo_chi2 = p
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
            [
                revenue_data[c]["total"] - revenue_data[c]["supportive"]
                for c in sorted(revenue_data)
            ],
        ]
        chi2, p, _, _ = chi2_contingency(table)
        rev_chi2 = p
        lines.append(f"Chi-squared test for revenue range: p={p:.3f}")
    elif chi2_contingency:
        lines.append("Not enough categories for chi-squared test of revenue range.")

    lines.append("\nSupport by CB rank:")
    for cat in sorted(rank_data):
        d = rank_data[cat]
        lines.append(f"  {cat}: {d['supportive']}/{d['total']} supportive")
    if chi2_contingency and len(rank_data) > 1:
        table = [
            [rank_data[c]["supportive"] for c in sorted(rank_data)],
            [
                rank_data[c]["total"] - rank_data[c]["supportive"]
                for c in sorted(rank_data)
            ],
        ]
        chi2, p, _, _ = chi2_contingency(table)
        rank_chi2 = p
        lines.append(f"Chi-squared test for CB rank: p={p:.3f}")
    elif chi2_contingency:
        lines.append("Not enough categories for chi-squared test of CB rank.")

    if size_records:
        size_vals_support = []
        size_vals_non = []
        emp_min, emp_max = (min(emp_vals), max(emp_vals)) if emp_vals else (None, None)
        rev_min, rev_max = (min(rev_vals), max(rev_vals)) if rev_vals else (None, None)
        rank_min, rank_max = (
            (min(rank_vals), max(rank_vals)) if rank_vals else (None, None)
        )

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
                tstat, pval = ttest_ind(
                    size_vals_support, size_vals_non, equal_var=False
                )
                size_ttest = (tstat, pval)
                lines.append(
                    f"T-test comparing size metric: t={tstat:.2f}, p={pval:.3f}"
                )
            elif ttest_ind:
                lines.append("Not enough data for t-test of size metric.")

    industries_all = [ind for c in companies for ind in _industry_list(c)]
    ipo_statuses = [c.ipo_status or "Unknown" for c in companies]
    emp_values = [e for c in companies if (e := _employee_count(c)) is not None]

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

    if justifications:
        biz_iter_all = is_business_flags if is_business_flags is not None else [None] * len(companies)
        examples = [
            (c.organization_name, p, j)
            for c, p, j, biz in zip(companies, percentiles, justifications, biz_iter_all)
            if j and biz is not False and _is_business(c.organization_name)
        ]
        if examples:
            # Select examples spread across the stance range [0, 1]
            targets = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            chosen = []
            used = set()
            for t in targets:
                best = None
                best_diff = None
                best_idx = None
                for idx, (_n, s, _j) in enumerate(examples):
                    if idx in used or s is None:
                        continue
                    diff = abs(s - t)
                    if best_diff is None or diff < best_diff:
                        best = examples[idx]
                        best_diff = diff
                        best_idx = idx
                if best is not None:
                    used.add(best_idx)
                    chosen.append(best)

            if chosen:
                lines.append("\nExample justifications:")
                for name, stance, justif in chosen:
                    if stance is None:
                        label = "Unknown"
                    else:
                        label = "Support" if stance >= 0.5 else "Oppose"
                    lines.append(f"  {name} ({label}): {justif}")

    lines.append("\nConclusions:")
    support_pct = (total_support / total_companies * 100) if total_companies else 0.0
    mc_ind, mc_count = (ind_name, ind_count) if industries_all else ("Unknown", 0)
    mc_support = industry_data[mc_ind]["supportive"] if mc_ind in industry_data else 0
    mc_pct = (mc_support / mc_count * 100) if mc_count else 0.0
    paragraph_parts = [
        f"Among the {total_companies} companies analyzed, {total_support} ({support_pct:.1f}%) were supportive of interoperability." ,
        f"The most represented industry was {mc_ind} with {mc_count} firms, of which {mc_pct:.1f}% were supportive."
    ]
    if emp_ttest:
        tstat, pval = emp_ttest
        significance = "significantly" if pval < 0.05 else "not significantly"
        paragraph_parts.append(
            f"Employee counts suggest supportive companies are {significance} smaller (t={tstat:.2f}, p={pval:.3f})."
        )
    if size_ttest:
        tstat, pval = size_ttest
        significance = "significant" if pval < 0.05 else "no significant"
        paragraph_parts.append(
            f"A derived size metric shows {significance} difference (t={tstat:.2f}, p={pval:.3f})."
        )
    if ipo_chi2 is not None:
        significance = "significant" if ipo_chi2 < 0.05 else "no significant"
        paragraph_parts.append(
            f"IPO status shows {significance} association with support (p={ipo_chi2:.3f})."
        )
    if rev_chi2 is not None:
        significance = "significant" if rev_chi2 < 0.05 else "no significant"
        paragraph_parts.append(
            f"Revenue range shows {significance} association (p={rev_chi2:.3f})."
        )
    if rank_chi2 is not None:
        significance = "significant" if rank_chi2 < 0.05 else "no significant"
        paragraph_parts.append(
            f"CB rank has {significance} association (p={rank_chi2:.3f})."
        )
    lines.append(" ".join(paragraph_parts))

    return "\n".join(lines)


async def _make_client():
    """Return an OpenAI client compatible with old and new libraries.

    Raises
    ------
    EnvironmentError
        If ``OPENAI_API_KEY`` is not defined in the environment.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")

    if hasattr(openai, "AsyncOpenAI"):
        return openai.AsyncOpenAI(api_key=api_key)
    return openai.OpenAI(api_key=api_key)


async def _collect_company_data(
    companies,
    max_concurrency: int,
    model_name: str,
) -> tuple[
    List[Optional[float]],
    List[Optional[str]],
    List[Optional[str]],
    List[Optional[bool]],
    List[List[str]],
    int,
]:
    """Fetch and parse web info for each company."""

    # Import ``async_fetch_company_web_info`` directly from ``company_lookup``
    # rather than through ``lookup_companies`` so tests can patch the function
    # at its canonical location.
    from company_lookup import async_fetch_company_web_info

    semaphore = asyncio.Semaphore(max_concurrency)
    client = await _make_client()

    stances: List[Optional[float]] = []
    subcats: List[Optional[str]] = []
    just_list: List[Optional[str]] = []
    biz_list: List[Optional[bool]] = []
    cached_count = 0
    table_rows: List[List[str]] = []

    async def fetch(company):
        async with semaphore:
            return await async_fetch_company_web_info(
                company.organization_name,
                model=model_name,
                return_cache_info=True,
                client=client,
            )

    tasks = [asyncio.create_task(fetch(c)) for c in companies]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for company, result in zip(companies, results):
        stance_val: Optional[float] = None
        subcat: Optional[str] = None
        justification: Optional[str] = None
        is_biz: Optional[bool] = None
        summary_text = ""
        if isinstance(result, Exception):
            pass
        elif result:
            content, cached = result
            if cached:
                cached_count += 1
            if content:
                print(content)
                parsed = parse_llm_response(content)
                if parsed is not None:
                    stance_val = parsed.get("supportive")
                    justification = parsed.get("justification")
                    subcat = parsed.get("sub_category")
                    parsed_summary = (
                        parsed.get("business_model_summary")
                        or parsed.get("business_model")
                        or parsed.get("summary")
                    )
                    is_biz = parsed.get("is_business")
                else:
                    parsed_summary = None

                summary_text = re.split(
                    r"```(?:json)?\s*\{.*?\}\s*```", content, flags=re.DOTALL
                )[0].strip()
                if not summary_text:
                    summary_text = parsed_summary or ""
        stance_label: str
        rank_str: str
        if stance_val is None:
            stance_label = "Unknown"
            rank_str = ""
        else:
            stance_label = "Support" if stance_val >= 0.5 else "Oppose"
            rank_str = f"{stance_val:.2f}"

        stances.append(stance_val)
        subcats.append(subcat)
        just_list.append(justification)
        biz_list.append(is_biz)
        table_rows.append(
            [
                company.organization_name,
                _industry(company),
                subcat or "",
                summary_text,
                stance_label,
                justification or summary_text,
                rank_str,
            ]
        )

    # Replace raw stance scores with percentile ranks for the CSV output
    ranks = percentile_ranks(stances)
    for idx, r in enumerate(ranks):
        table_rows[idx][6] = "" if r is None else f"{r:.2f}"
        if r is None:
            table_rows[idx][4] = "Unknown"
        else:
            table_rows[idx][4] = "Support" if r >= 0.5 else "Oppose"

    close_method = getattr(client, "aclose", None)
    if close_method is None:
        close_method = getattr(client, "close", None)
    if close_method:
        try:
            result = close_method()
            if inspect.isawaitable(result):
                await result
        except Exception:
            pass

    return stances, subcats, just_list, biz_list, table_rows, cached_count


async def run_async(
    companies,
    max_concurrency: int,
    output_dir: Path,
    *,
    model_name: str = "gpt-4o",
) -> None:

    (
        stances,
        subcats,
        just_list,
        biz_list,
        table_rows,
        cached_count,
    ) = await _collect_company_data(companies, max_concurrency, model_name)

    report = generate_final_report(
        companies,
        stances,
        subcats,
        just_list,
        biz_list,
    )
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
    abstract = await async_report_to_abstract(report, model=model_name)
    abstract_path = output_dir / "abstract.txt"
    abstract_path.write_text(abstract or "", encoding="utf-8")
    print(f"Output table saved to {table_path}")
    print(f"Report saved to {report_path}")
    print(f"Abstract saved to {abstract_path}")

