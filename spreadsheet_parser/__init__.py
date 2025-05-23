"""Utility package for parsing company spreadsheets and looking up company info."""

import logging

logging.basicConfig(level=logging.INFO)

from .models import Company, LLMOutput
from .csv_reader import (
    read_companies_from_csv,
    read_companies_from_xlsx,
    read_companies_from_csvs,
)
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
    DEFAULT_MAX_INDUSTRIES,
    generate_final_report,
    percentile_ranks,
    run_async,
    _industry,
    _industry_list,
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
    "LLMOutput",
    "read_companies_from_csv",
    "read_companies_from_csvs",
    "read_companies_from_xlsx",
    "fetch_company_web_info",
    "async_fetch_company_web_info",
    "parse_llm_response",
    "report_to_abstract",
    "async_report_to_abstract",
    "DEFAULT_MAX_LINES",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_MAX_INDUSTRIES",
    "generate_final_report",
    "percentile_ranks",
    "run_async",
    "_run_async",
    "_industry",
    "_industry_list",
    "_employee_count",
    "_ipo_category",
    "_revenue_value",
    "_revenue_category",
    "_cb_rank_value",
    "_cb_rank_category",
]
