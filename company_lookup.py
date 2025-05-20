import ast
import asyncio
import hashlib
import inspect
import json
import os
import re
from dataclasses import asdict
from parser import Company
from pathlib import Path
from typing import Optional, Tuple, Union

import openai

# AI industry sub-category taxonomy derived from the dataset
_AI_SUBCATEGORIES = [
    "Generative AI",
    "Machine Learning",
    "Natural Language Processing",
    "Computer Vision",
    "Robotics",
    "AI Infrastructure",
    "AI Hardware",
    "Big Data & Analytics",
    "Security AI",
    "Finance AI",
    "Health/Bio AI",
    "Education AI",
    "Marketing AI",
    "Other AI",
    "Non-AI",
]


async def _fetch_with_cache(
    company: Union[str, Company],
    client,
    model_name: str,
    seed: Optional[int],
    return_cache_info: bool,
) -> Union[Optional[str], Tuple[Optional[str], bool]]:
    """Shared helper to build the prompt, hit the cache and call the API."""

    if isinstance(company, str):
        company_name = company
        csv_json = ""
    else:
        company_name = company.organization_name
        csv_json = json.dumps(asdict(company), ensure_ascii=False, indent=2)

    prompt = f"Search the web for information about {company_name}. "
    if csv_json:
        prompt += (
            "Here is the original spreadsheet row as JSON. "
            "Correct any malformed values and respond with the cleaned data.\n"
            "```json\n" + csv_json + "\n```\n"
        )
    taxonomy_list = "; ".join(_AI_SUBCATEGORIES)
    prompt += (
        "Then summarize the company's business model and data strategy and "
        "rate their likely support for interoperability legislation on a scale "
        "from 0 (strong opponent) to 1 (strong proponent). "
        "Mozilla and the Electronic Frontier Foundation would be close to 1, "
        "while Meta and Palantir might be near 0. "
        "Classify the company using one of these AI sub-categories: "
        f"{taxonomy_list}. "
        "Return ONLY a JSON code block containing the sanitized fields along "
        "with a 'sub_category' string and numeric 'supportive' value."
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

    kwargs = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an assistant that can access web search results "
                    "to provide up-to-date company information."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    if seed is not None:
        kwargs["seed"] = seed

    response = client.chat.completions.create(**kwargs)
    if inspect.isawaitable(response):
        response = await response

    message = response.choices[0].message
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)

    if content is not None:
        cache_file.write_text(content, encoding="utf-8")
    return (content, False) if return_cache_info else content


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

    return asyncio.run(
        _fetch_with_cache(
            company,
            client,
            model_name,
            seed,
            return_cache_info,
        )
    )


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

    return await _fetch_with_cache(
        company,
        client,
        model_name,
        seed,
        return_cache_info,
    )


def _parse_support_value(value: object) -> Optional[float]:
    """Normalize the ``supportive`` field to a float between 0 and 1."""

    if isinstance(value, (int, float)):
        num = float(value)
        if 0.0 <= num <= 1.0:
            return num
        return None

    text = str(value).strip().lower()
    if text in {"true", "yes"}:
        return 1.0
    if text in {"false", "no"}:
        return 0.0

    match_num = re.search(r"-?\d*\.?\d+", text)
    if match_num:
        try:
            num = float(match_num.group())
            if 0.0 <= num <= 1.0:
                return num
        except ValueError:
            pass
    return None


def parse_llm_response(response: str) -> Optional[dict]:
    """Extract sanitized fields and the ``supportive`` value from the LLM response."""

    if not response:
        return None

    match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", response, re.IGNORECASE | re.DOTALL
    )
    if not match:
        return None

    json_text = match.group(1).strip()
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(json_text)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None

    supportive = _parse_support_value(data.get("supportive"))
    data["supportive"] = supportive

    subcat = data.get("sub_category") or data.get("subcategory")
    if isinstance(subcat, str):
        data["sub_category"] = subcat.strip()
    else:
        data["sub_category"] = None

    return data
