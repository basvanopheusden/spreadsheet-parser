import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class Company:
    organization_name: str
    organization_name_url: Optional[str]
    estimated_revenue_range: Optional[str]
    ipo_status: Optional[str]
    operating_status: Optional[str]
    acquisition_status: Optional[str]
    company_type: Optional[str]
    number_of_employees: Optional[str]
    full_description: Optional[str]
    industries: Optional[str]
    headquarters_location: Optional[str]
    description: Optional[str]
    cb_rank: Optional[str]


_CANONICAL_HEADERS: Dict[str, str] = {
    "organization_name": "Organization Name",
    "organization_name_url": "Organization Name URL",
    "estimated_revenue_range": "Estimated Revenue Range",
    "ipo_status": "IPO Status",
    "operating_status": "Operating Status",
    "acquisition_status": "Acquisition Status",
    "company_type": "Company Type",
    "number_of_employees": "Number of Employees",
    "full_description": "Full Description",
    "industries": "Industries",
    "headquarters_location": "Headquarters Location",
    "description": "Description",
    "cb_rank": "CB Rank (Company)",
}


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

        sanitized_map = {_sanitize(v): k for k, v in _CANONICAL_HEADERS.items()}
        field_map: Dict[str, str] = {}
        for header in reader.fieldnames or []:
            key = sanitized_map.get(_sanitize(header))
            if key:
                field_map[key] = header

        for row in reader:
            kwargs = {}
            for attr in _CANONICAL_HEADERS.keys():
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
