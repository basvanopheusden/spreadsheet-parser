from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import csv


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


def read_companies_from_csv(path: Union[str, Path]) -> List[Company]:
    """Read company data from a CSV file and return a list of Company objects."""
    path = Path(path)
    companies: List[Company] = []
    with path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            company = Company(
                organization_name=row.get("Organization Name", ""),
                organization_name_url=row.get("Organization Name URL"),
                estimated_revenue_range=row.get("Estimated Revenue Range"),
                ipo_status=row.get("IPO Status"),
                operating_status=row.get("Operating Status"),
                acquisition_status=row.get("Acquisition Status"),
                company_type=row.get("Company Type"),
                number_of_employees=row.get("Number of Employees"),
                full_description=row.get("Full Description"),
                industries=row.get("Industries"),
                headquarters_location=row.get("Headquarters Location"),
                description=row.get("Description"),
                cb_rank=row.get("CB Rank (Company)"),
            )
            companies.append(company)
    return companies
