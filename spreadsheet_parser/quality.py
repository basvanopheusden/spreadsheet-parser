import logging
from typing import Iterable, List, Sequence

from .analysis import _employee_count, _revenue_value

from .models import Company

logger = logging.getLogger(__name__)


def find_duplicate_names(companies: Sequence[Company]) -> List[str]:
    """Return a sorted list of organization names that appear more than once."""
    seen = set()
    duplicates = set()
    for company in companies:
        name = company.organization_name.strip().lower()
        if name in seen:
            duplicates.add(company.organization_name)
        else:
            seen.add(name)
    return sorted(duplicates)


def rows_with_missing_fields(
    companies: Sequence[Company], fields: Iterable[str]
) -> List[int]:
    """Return indexes of rows missing any of the specified fields."""
    missing = []
    for idx, company in enumerate(companies):
        for field in fields:
            value = getattr(company, field, None)
            if value is None or value == "":
                logger.warning("Row %d missing %s", idx, field)
                missing.append(idx)
                break
    return missing


def revenue_employee_outliers(
    companies: Sequence[Company],
    *,
    employee_threshold: int = 50,
    revenue_threshold: float = 1000.0,
) -> List[int]:
    """Return indexes where revenue is implausibly high for employee counts.

    Parameters
    ----------
    companies:
        Sequence of :class:`~spreadsheet_parser.models.Company` objects.
    employee_threshold:
        Maximum employee count considered "small".
    revenue_threshold:
        Minimum revenue in millions considered suspicious.
    """

    outliers: List[int] = []
    for idx, company in enumerate(companies):
        emp = _employee_count(company)
        rev = _revenue_value(company.estimated_revenue_range)
        if (
            emp is not None
            and rev is not None
            and emp <= employee_threshold
            and rev >= revenue_threshold
        ):
            logger.warning(
                "Row %d unlikely revenue %s for %s employees",
                idx,
                company.estimated_revenue_range,
                company.number_of_employees,
            )
            outliers.append(idx)
    return outliers


def rows_with_garbled_text(
    companies: Sequence[Company], fields: Iterable[str] | None = None
) -> List[int]:
    """Return indexes of rows with suspiciously garbled text in ``fields``."""

    if fields is None:
        fields = ["industries", "headquarters_location", "description"]

    garbled: List[int] = []
    for idx, company in enumerate(companies):
        for field in fields:
            value = getattr(company, field, None)
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            for val in values:
                text = str(val)
                if (
                    len(text) > 100
                    or "http://" in text
                    or "https://" in text
                    or "www." in text
                    or "\n" in text
                    or text.count(":") > 1
                ):
                    logger.warning("Row %d garbled %s: %s", idx, field, text)
                    garbled.append(idx)
                    break
            if garbled and garbled[-1] == idx:
                break
    return garbled
