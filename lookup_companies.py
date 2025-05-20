from pathlib import Path
from argparse import ArgumentParser
import asyncio
import re

from spreadsheet_parser.analysis import (
    run_async,
    generate_final_report,
    DEFAULT_MAX_LINES,
    DEFAULT_MAX_CONCURRENCY,
    _industry,
)
from spreadsheet_parser.csv_reader import read_companies_from_csv

__all__ = [
    "run_async",
    "generate_final_report",
    "DEFAULT_MAX_LINES",
    "DEFAULT_MAX_CONCURRENCY",
    "_industry",
    "main",
]

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
