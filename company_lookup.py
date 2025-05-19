import os
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union, Tuple
import json
import re

from parser import Company

import openai


def fetch_company_web_info(
    company: Union[str, Company],
    model: Optional[str] = None,
    *,
    seed: Optional[int] = None,
    return_cache_info: bool = False,
) -> Union[Optional[str], Tuple[Optional[str], bool]]:
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

    if seed is None:
        env_seed = os.getenv("OPENAI_SEED")
        if env_seed is not None:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = None

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
        "stance on interoperability and access legislation. "
        "Rate their support on a scale from 0 (strong opponent) to 1 (strong proponent). "
        "End with a JSON code block containing the key 'supportive' with this number. "
        "For example:\n"
        "```json\n{\"supportive\": 0.8}\n``` "
        "Mozilla and the Electronic Frontier Foundation would be close to 1, "
        "while Meta and Palantir might be near 0. "
        "Finish with ONLY the JSON block on a new line."

    )

    cache_dir = Path.home() / "llm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(
        f"{model_name}\n{prompt}\n{seed}".encode("utf-8")
    ).hexdigest()
    cache_file = cache_dir / f"{cache_key}.txt"
    if cache_file.exists():
        content = cache_file.read_text(encoding="utf-8")
        return (content, True) if return_cache_info else content

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
        **({"seed": seed} if seed is not None else {}),
    )

    message = response.choices[0].message
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)

    if content is not None:
        cache_file.write_text(content, encoding="utf-8")
    return (content, False) if return_cache_info else content


async def async_fetch_company_web_info(
    company: Union[str, Company],
    model: Optional[str] = None,
    *,
    seed: Optional[int] = None,
    return_cache_info: bool = False,
) -> Union[Optional[str], Tuple[Optional[str], bool]]:
    """Asynchronously fetch company info using OpenAI.

    This mirrors :func:`fetch_company_web_info` but operates asynchronously and
    uses the same file-based cache to avoid duplicate API calls.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")

    client = openai.AsyncOpenAI(api_key=api_key)

    model_name = model or os.getenv("OPENAI_MODEL") or "gpt-4o"

    if seed is None:
        env_seed = os.getenv("OPENAI_SEED")
        if env_seed is not None:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = None

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
        "stance on interoperability and access legislation. "
        "Rate their support on a scale from 0 (strong opponent) to 1 (strong proponent). "
        "End with a JSON code block containing the key 'supportive' with this number. "
        "For example:\n"
        "```json\n{\"supportive\": 0.8}\n``` "
        "Mozilla and the Electronic Frontier Foundation would be close to 1, "
        "while Meta and Palantir might be near 0. "
        "Finish with ONLY the JSON block on a new line."
    )

    cache_dir = Path.home() / "llm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(
        f"{model_name}\n{prompt}\n{seed}".encode("utf-8")
    ).hexdigest()
    cache_file = cache_dir / f"{cache_key}.txt"
    if cache_file.exists():
        content = cache_file.read_text(encoding="utf-8")
        return (content, True) if return_cache_info else content

    response = await client.chat.completions.create(
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
        **({"seed": seed} if seed is not None else {}),
    )

    message = response.choices[0].message
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)

    if content is not None:
        cache_file.write_text(content, encoding="utf-8")
    return (content, False) if return_cache_info else content


def parse_llm_response(response: str) -> Optional[float]:
    """Extract the numeric `supportive` value from the JSON markdown block in the LLM response."""

    if not response:
        return None

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.IGNORECASE | re.DOTALL)
    if not match:
        return None

    try:
        json_text = match.group(1).replace("'", '"')
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return None

    supportive = data.get("supportive")
    if supportive is None:
        return None

    if isinstance(supportive, (int, float)):
        value = float(supportive)
        if 0.0 <= value <= 1.0:
            return value
        return None

    text = str(supportive).strip().lower()
    if text in {"true", "yes"}:
        return 1.0
    if text in {"false", "no"}:
        return 0.0

    match_num = re.search(r"-?\d*\.?\d+", text)
    if match_num:
        try:
            value = float(match_num.group())
            if 0.0 <= value <= 1.0:
                return value
        except ValueError:
            pass
    return None

