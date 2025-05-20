import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Union

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


_MONTH_MAP = {
    "Jan": "1",
    "Feb": "2",
    "Mar": "3",
    "Apr": "4",
    "May": "5",
    "Jun": "6",
    "Jul": "7",
    "Aug": "8",
    "Sep": "9",
    "Oct": "10",
    "Nov": "11",
    "Dec": "12",
}


def _decode_lines(file_obj) -> Iterable[str]:
    """Yield decoded lines from a binary file object with fallback."""

    def decode(b: bytes, *, first: bool = False) -> str:
        enc = "utf-8-sig" if first else "utf-8"
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            return b.decode("latin-1").encode("utf-8", "replace").decode("utf-8")

    first = True
    for bline in file_obj:
        yield decode(bline, first=first)
        first = False


def _fix_employee_range(text: str) -> str:
    """Normalize malformed employee range strings."""

    m = re.match(r"^(\d+)-([A-Za-z]{3})$", text)
    if m:
        day, mon = m.groups()
        mon_num = _MONTH_MAP.get(mon.title())
        if mon_num:
            return f"{mon_num}-{day}"

    m = re.match(r"^([A-Za-z]{3})-(\d+)$", text)
    if m:
        mon, year = m.groups()
        mon_num = _MONTH_MAP.get(mon.title())
        if mon_num:
            return f"{mon_num}-{year}"

    return text


def _is_business(name: str) -> bool:
    """Return True if the name appears to describe a business."""
    text = name.lower()
    return not any(re.search(pat, text) for pat in _EXCLUDE_PATTERNS)


def read_companies_from_csv(path: Union[str, Path]) -> List[Company]:
    """Read company data from a CSV file and return a list of Company objects."""

    path = Path(path)
    companies: List[Company] = []

    with path.open("rb") as raw:
        sample_bytes = raw.read(2048)
        raw.seek(0)
        try:
            sample = sample_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            sample = sample_bytes.decode("latin-1").encode("utf-8", "replace").decode("utf-8")

        try:
            dialect = csv.Sniffer().sniff(sample)
            if dialect.delimiter not in {",", ";", "\t", "|"}:
                raise csv.Error("unlikely delimiter")
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(_decode_lines(raw), dialect=dialect)

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
                elif attr == "number_of_employees" and value is not None:
                    kwargs[attr] = _fix_employee_range(value) if value != "" else None
                else:
                    kwargs[attr] = value if value != "" else None

            company = Company(**kwargs)
            if not _is_business(company.organization_name):
                continue
            companies.append(company)

    return companies


def read_companies_from_xlsx(path: Union[str, Path]) -> List[Company]:
    """Read company data from an XLSX file and return a list of ``Company`` objects.

    This function requires the optional :mod:`openpyxl` package. If it is not
    installed, an :class:`ImportError` is raised with a helpful message.
    """

    try:
        import openpyxl  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "openpyxl is required to read .xlsx files"
        ) from exc

    path = Path(path)
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active

    header_cells = [str(c.value or "") for c in next(ws.iter_rows(min_row=1, max_row=1))]

    sanitized_map = {_sanitize(v): k for k, v in CANONICAL_HEADERS.items()}
    field_map: Dict[str, int] = {}
    for idx, header in enumerate(header_cells):
        key = sanitized_map.get(_sanitize(header))
        if key:
            field_map[key] = idx

    companies: List[Company] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        kwargs = {}
        for attr in CANONICAL_HEADERS.keys():
            idx = field_map.get(attr)
            value = row[idx] if idx is not None and idx < len(row) else None
            if isinstance(value, str):
                value = value.strip()
            if attr == "organization_name":
                kwargs[attr] = value or ""
            else:
                kwargs[attr] = value if value not in (None, "") else None

        company = Company(**kwargs)
        if not _is_business(company.organization_name):
            continue
        companies.append(company)

    return companies
