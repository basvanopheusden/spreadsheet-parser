import csv
from pathlib import Path
from typing import Dict, List, Union

from .models import Company, CANONICAL_HEADERS


def _sanitize(text: str) -> str:
    """Normalize header names for matching."""
    return "".join(ch.lower() for ch in text if ch.isalnum())


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

            companies.append(Company(**kwargs))

    return companies
