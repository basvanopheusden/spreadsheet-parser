import os
from dataclasses import asdict
from typing import Optional, Union

from parser import Company

import openai


def fetch_company_web_info(
    company: Union[str, Company], model: Optional[str] = None
) -> Optional[str]:
    """Ask an LLM to search the web for company information.

    This stub assumes the model can perform web searches. It sends a prompt to
    the OpenAI API using the ``OPENAI_API_KEY`` environment variable. Callers
    may provide either a company name or a :class:`Company` instance obtained
    from :func:`parser.read_companies_from_csv`. If a dataclass is given, all
    available details are provided to the model.  The default model targets a
    modern GPT-4 version, but callers may override it via the ``model``
    parameter or ``OPENAI_MODEL`` environment variable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")

    client = openai.OpenAI(api_key=api_key)

    model_name = model or os.getenv("OPENAI_MODEL") or "gpt-4o"

    if isinstance(company, str):
        company_name = company
        csv_details = ""
    else:
        company_name = company.organization_name
        info = asdict(company)
        info.pop("organization_name", None)
        lines = [f"{k.replace('_', ' ').title()}: {v}" for k, v in info.items() if v]
        csv_details = "\n".join(lines)

    prompt = f"Search the web for information about {company_name}. "
    if csv_details:
        prompt += "Here is what we already know from a CSV:\n" + csv_details + "\n"
    prompt += (
        "Summarize the company's business model, data strategy, and likely "
        "stance on interoperability and access legislation."
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant that can access web search results "
                    "to provide up-to-date company information."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.get("content")

