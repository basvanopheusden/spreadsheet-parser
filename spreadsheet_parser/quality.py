import logging
from typing import Iterable, List, Sequence

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
