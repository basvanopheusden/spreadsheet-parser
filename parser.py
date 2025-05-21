"""Thin compatibility layer that re-exports spreadsheet parsing helpers."""

from spreadsheet_parser import (
    Company,
    LLMOutput,
    read_companies_from_csv,
    read_companies_from_csvs,
    read_companies_from_xlsx,
)

__all__ = [
    "Company",
    "LLMOutput",
    "read_companies_from_csv",
    "read_companies_from_csvs",
    "read_companies_from_xlsx",
]
