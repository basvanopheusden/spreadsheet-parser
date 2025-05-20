import csv
import re
from pathlib import Path
from typing import Dict, List, Union

from .models import Company, CANONICAL_HEADERS


def _sanitize(text: str) -> str:
    """Normalize header names for matching."""
    return "".join(ch.lower() for ch in text if ch.isalnum())


_EXCLUDE_PATTERNS = [
    r"\binstitute\b",
    r"\buniversity\b",
    r"\bcollege\b",
    r"\bfoundation\b",
    r"\bcentre\b",
    r"\bcenter\b",
]


def _is_business(name: str) -> bool:
    """Return True if the name appears to describe a business."""
    text = name.lower()
    return not any(re.search(pat, text) for pat in _EXCLUDE_PATTERNS)


def read_companies_from_csv(path: Union[str, Path]) -> List[Company]:
    """Read company data from a CSV file and return a list of Company objects."""

    path = Path(path)
    companies: List[Company] = []

    with path.open(newline="", encoding="utf-8-sig") as csvfile:
        sample = csvfile.read(2048)
        csvfile.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(csvfile, dialect=dialect)

        sanitized_map = {_sanitize(v): k for k, v in CANONICAL_HEADERS.items()}
        field_map: Dict[str, str] = {}
        for header in reader.fieldnames or []:
            key = sanitized_map.get(_sanitize(header))
            if key:
                field_map[key] = header

        for row in reader:
            kwargs = {}
            for attr in CANONICAL_HEADERS.keys():
                header = field_map.get(attr)
                value = row.get(header) if header else None
                if value is not None:
                    value = value.strip()
                if attr == "organization_name":
                    kwargs[attr] = value or ""
                else:
                    kwargs[attr] = value if value != "" else None

            company = Company(**kwargs)
            if not _is_business(company.organization_name):
                continue
            companies.append(company)

    return companies
