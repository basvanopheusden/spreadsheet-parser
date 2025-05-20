from pathlib import Path
from argparse import ArgumentParser
import asyncio
import csv
import re

from spreadsheet_parser import (
    async_fetch_company_web_info,
    run_async,
    generate_final_report,
    DEFAULT_MAX_LINES,
    DEFAULT_MAX_CONCURRENCY,
    _industry,
    parse_llm_response,
)
from spreadsheet_parser.csv_reader import read_companies_from_csv
from spreadsheet_parser.quality import find_duplicate_names, rows_with_missing_fields

__all__ = [
    "run_async",
    "_run_async",
    "generate_final_report",
    "DEFAULT_MAX_LINES",
    "async_fetch_company_web_info",
    "DEFAULT_MAX_CONCURRENCY",
    "_industry",
    "main",
]


async def _run_async(companies, max_concurrency: int, output_dir: Path) -> None:
    semaphore = asyncio.Semaphore(max_concurrency)

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
                return_cache_info=True,
            )

    tasks = [asyncio.create_task(fetch(c)) for c in companies]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for company, result in zip(companies, results):
        if isinstance(result, Exception):
            stances.append(None)
            subcats.append(None)
            just_list.append(None)
            table_rows.append(
                [
                    company.organization_name,
                    _industry(company),
                    "",
                    "",
                    "Unknown",
                    "",
                    "",
                ]
            )
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
                    parsed_summary = None
                    is_biz = None
                else:
                    stance_val = parsed.get("supportive")
                    justification = parsed.get("justification")
                    subcat = parsed.get("sub_category")
                    parsed_summary = (
                        parsed.get("business_model_summary")
                        or parsed.get("business_model")
                        or parsed.get("summary")
                    )
                    is_biz = parsed.get("is_business")

                stances.append(stance_val)
                subcats.append(subcat)
                just_list.append(justification)
                biz_list.append(is_biz)

                summary_text = re.split(
                    r"```(?:json)?\s*\{.*?\}\s*```", content, flags=re.DOTALL
                )[0].strip()
                if not summary_text:
                    summary_text = parsed_summary or ""

                if stance_val is None:
                    stance_label = "Unknown"
                    rank_str = ""
                else:
                    stance_label = "Support" if stance_val >= 0.5 else "Oppose"
                    rank_str = f"{stance_val:.2f}"

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
            else:
                stances.append(None)
                subcats.append(None)
                just_list.append(None)
                biz_list.append(None)
                table_rows.append(
                    [
                        company.organization_name,
                        _industry(company),
                        "",
                        "",
                        "Unknown",
                        "",
                        "",
                    ]
                )

        else:
            stances.append(None)
            subcats.append(None)
            just_list.append(None)
            biz_list.append(None)
            table_rows.append(
                [
                    company.organization_name,
                    _industry(company),
                    "",
                    "",
                    "Unknown",
                    "",
                    "",
                ]
            )

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
    print(f"Output table saved to {table_path}")
    print(f"Report saved to {report_path}")


def main() -> None:
    parser = ArgumentParser(
        description="Fetch web summaries for companies listed in a CSV file"
    )
    parser.add_argument("csv", type=Path, help="CSV file containing company data")
    parser.add_argument(
        "--max-lines",
        type=int,
        default=DEFAULT_MAX_LINES,
        help=f"Number of lines to process (default: {DEFAULT_MAX_LINES})",
    )
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
    duplicates = find_duplicate_names(companies)
    if duplicates:
        print("Duplicate organization names found:")
        for name in duplicates:
            print(f"  {name}")

    rows_with_missing_fields(companies, ["organization_name_url"])
    total_rows = len(companies)

    if total_rows > 100:
        print(
            f"Warning: {args.csv} contains {total_rows} rows. "
            f"Only the first {args.max_lines} will be processed."
        )

    to_process = companies[: args.max_lines]

    asyncio.run(run_async(to_process, args.max_concurrency, args.output_dir))


if __name__ == "__main__":
    main()
