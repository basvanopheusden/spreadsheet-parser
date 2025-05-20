"""CLI tool for processing company spreadsheets and generating LLM-based reports."""

from pathlib import Path
from argparse import ArgumentParser
import asyncio
import csv
import os

from spreadsheet_parser.analysis import (
    generate_final_report,
    DEFAULT_MAX_LINES,
    DEFAULT_MAX_CONCURRENCY,
    _industry,
    _collect_company_data,
)
from spreadsheet_parser.llm import async_report_to_abstract
from company_lookup import async_fetch_company_web_info
from spreadsheet_parser.csv_reader import (
    read_companies_from_csv,
    read_companies_from_xlsx,
)

from spreadsheet_parser.quality import find_duplicate_names, rows_with_missing_fields

__all__ = [
    "run_async",
    "generate_final_report",
    "DEFAULT_MAX_LINES",
    "DEFAULT_MAX_CONCURRENCY",
    "_industry",
    "main",
]

async def _run_async(
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
        plot_path=output_dir / "support_by_subcat.png",
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

# Expose the async runner for tests
run_async = _run_async

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
    parser.add_argument(
        "--model-name",
        default="gpt-4o",
        help="OpenAI model to use when fetching web summaries",
    )
    args = parser.parse_args()

    if args.csv.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
        companies = read_companies_from_xlsx(args.csv)
    else:
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

    asyncio.run(
        run_async(
            to_process,
            args.max_concurrency,
            args.output_dir,
            model_name=args.model_name,
        )
    )


if __name__ == "__main__":
    main()
