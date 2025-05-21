"""CLI tool for processing company spreadsheets and generating LLM-based reports."""

from pathlib import Path
from argparse import ArgumentParser
import asyncio
import csv

from dataclasses import asdict
from spreadsheet_parser.analysis import (
    generate_final_report,
    DEFAULT_MAX_LINES,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_INDUSTRIES,
    _industry,
    _collect_company_data,
)
import spreadsheet_parser.analysis
from spreadsheet_parser.llm import async_report_to_abstract
from spreadsheet_parser.csv_reader import read_companies_from_csvs

from spreadsheet_parser.quality import find_duplicate_names, rows_with_missing_fields

__all__ = [
    "run_async",
    "generate_final_report",
    "DEFAULT_MAX_LINES",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_MAX_INDUSTRIES",
    "_industry",
    "main",
]

async def _run_async(
    companies,
    max_concurrency: int,
    output_dir: Path,
    *,
    model_name: str = "gpt-4o",
    quality_sample_size: int = 100,
    max_industries: int = DEFAULT_MAX_INDUSTRIES,
) -> None:

    quality_notes = await spreadsheet_parser.analysis._sample_data_quality_report(
        companies, model_name, sample_size=quality_sample_size
    )

    (
        stances,
        subcats,
        just_list,
        biz_list,
        mal_list,
        table_rows,
        cached_count,
        parsed_list,
    ) = await _collect_company_data(companies, max_concurrency, model_name)

    report = generate_final_report(
        companies,
        stances,
        subcats,
        just_list,
        biz_list,
        is_malformed_flags=mal_list,
        plot_path=output_dir / "support_by_subcat.png",
        max_industries=max_industries,

    )
    if quality_notes:
        report = "Data Quality Review:\n" + quality_notes.strip() + "\n\n" + report
    print(report)
    print(f"Cached responses used: {cached_count}")

    output_dir.mkdir(parents=True, exist_ok=True)
    malformed_names = [
        c.organization_name
        for c, mal in zip(companies, mal_list)
        if mal
    ]
    spreadsheet_parser.analysis._write_quality_report_csv(
        output_dir / "data_quality_report.csv",
        quality_notes,
        malformed_names,
    )
    spreadsheet_parser.analysis._write_quality_report_txt(
        output_dir / "data_quality_report.txt", quality_notes
    )
    malformed_rows = []
    for comp, parsed, flag in zip(companies, parsed_list, mal_list):
        if flag:
            row = asdict(comp)
            if parsed is not None:
                if parsed.raw:
                    row.update(parsed.raw)
                row.update({k: v for k, v in asdict(parsed).items() if k != "raw"})
            malformed_rows.append(row)
    if malformed_rows:
        spreadsheet_parser.analysis._write_malformed_data_csv(
            output_dir / "malformed_data.csv", malformed_rows
        )
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
        description="Fetch web summaries for companies listed in CSV files inside a directory"
    )
    parser.add_argument(
        "csv_dir",
        type=Path,
        help="Directory containing company CSV files",
    )
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
    parser.add_argument(
        "--quality-sample-size",
        type=int,
        default=100,
        help="Rows randomly sampled for the data quality report",
    )
    parser.add_argument(
        "--max-industries",
        type=int,
        default=DEFAULT_MAX_INDUSTRIES,
        help=f"Maximum industries shown in the final report (default: {DEFAULT_MAX_INDUSTRIES})",
    )
    args = parser.parse_args()

    csv_paths = [
        p
        for p in sorted(args.csv_dir.iterdir())
        if p.is_file() and p.suffix.lower() == ".csv"
    ]
    companies = read_companies_from_csvs(csv_paths)
    duplicates = find_duplicate_names(companies)
    if duplicates:
        print("Duplicate organization names found:")
        for name in duplicates:
            print(f"  {name}")

    rows_with_missing_fields(companies, ["organization_name_url"])
    total_rows = len(companies)

    if total_rows > 100:
        print(
            f"Warning: {args.csv_dir} contains {total_rows} rows. "
            f"Only the first {args.max_lines} will be processed."
        )

    to_process = companies[: args.max_lines]

    asyncio.run(
        run_async(
            to_process,
            args.max_concurrency,
            args.output_dir,
            model_name=args.model_name,
            quality_sample_size=args.quality_sample_size,
            max_industries=args.max_industries,
        )
    )


if __name__ == "__main__":
    main()
