"""Utility package for parsing company spreadsheets and looking up company info."""

from .models import Company
from .csv_reader import read_companies_from_csv, read_companies_from_xlsx
from .llm import (
    fetch_company_web_info,
    async_fetch_company_web_info,
    parse_llm_response,
    report_to_abstract,
    async_report_to_abstract,
)
from .analysis import (
    DEFAULT_MAX_LINES,
    DEFAULT_MAX_CONCURRENCY,
    generate_final_report,
    percentile_ranks,
    run_async,
    _industry,
    _employee_count,
    _ipo_category,
    _revenue_value,
    _revenue_category,
    _cb_rank_value,
    _cb_rank_category,
)

_run_async = run_async

__all__ = [
    "Company",
    "read_companies_from_csv",
    "read_companies_from_xlsx",
    "fetch_company_web_info",
    "async_fetch_company_web_info",
    "parse_llm_response",
    "report_to_abstract",
    "async_report_to_abstract",
    "DEFAULT_MAX_LINES",
    "DEFAULT_MAX_CONCURRENCY",
    "generate_final_report",
    "percentile_ranks",
    "run_async",
    "_run_async",
    "_industry",
    "_employee_count",
    "_ipo_category",
    "_revenue_value",
    "_revenue_category",
    "_cb_rank_value",
    "_cb_rank_category",
]
