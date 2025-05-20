from dataclasses import dataclass
from typing import Optional, Dict, List

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
    # ``Industries`` may contain multiple comma-separated values in the
    # original spreadsheet.  Store them as a list of strings so that a
    # company can be counted in more than one industry category.
    industries: Optional[List[str]]
    headquarters_location: Optional[str]
    description: Optional[str]
    cb_rank: Optional[str]


CANONICAL_HEADERS: Dict[str, str] = {
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
