from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple, Union

import openai

from .models import Company, LLMOutput


logger = logging.getLogger(__name__)


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
        "Indicate whether this is a legitimate for-profit business that earns "
        "revenue through selling products or services. If it is instead a "
        "research institute, advocacy organization, or other non-business "
        "corporation, mark it accordingly. "
        "Critically assess the company's public statements. They may claim "
        "strong values and support for interoperability, but business "
        "incentives could ultimately prevail. Predict the stance that best "
        "serves the company's interests. "
        "Return ONLY a JSON code block containing the sanitized fields along "
        "with a 'sub_category' string, numeric 'supportive' value, a boolean "
        "'is_business', a short 'business_model_summary', a brief "
        "'justification' for the rating, and a boolean "
        "'is_possibly_malformed' flag indicating if the input data might be "
        "corrupted or missing."
    )

    cache_dir = Path.home() / "llm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(
        f"{model_name}\n{prompt}\n{seed}".encode("utf-8")
    ).hexdigest()
    cache_file = cache_dir / f"{cache_key}.txt"
    if cache_file.exists():
        logger.info("Cache hit for %s", company_name)
        content = cache_file.read_text(encoding="utf-8")
        return (content, True) if return_cache_info else content
    else:
        logger.info("Cache miss for %s", company_name)

    input_text = (
        "You are an assistant that can access web search results "
        "to provide up-to-date company information.\n" + prompt
    )

    kwargs = {
        "model": model_name,
        "input": input_text,
        "tools": [{"type": "web_search_preview"}],
    }
    if seed is not None:
        kwargs["seed"] = seed

    try:
        response = client.responses.create(**kwargs)
        if inspect.isawaitable(response):
            response = await response
    except Exception:
        logger.exception("API request failed for %s", company_name)
        raise

    content = getattr(response, "output_text", None)

    if content is not None:
        cache_file.write_text(content, encoding="utf-8")
    return (content, False) if return_cache_info else content


def fetch_company_web_info(
    company: Union[str, Company],
    model: Optional[str] = None,
    *,
    seed: Optional[int] = None,
    return_cache_info: bool = False,
    client: Optional[openai.OpenAI] = None,
) -> Union[Optional[str], Tuple[Optional[str], bool]]:
    """Ask an LLM to search the web for company information."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")

    needs_close = False
    if client is None:
        client = openai.OpenAI(api_key=api_key)
        needs_close = True

    model_name = model or os.getenv("OPENAI_MODEL") or "gpt-4o"

    if seed is None:
        env_seed = os.getenv("OPENAI_SEED")
        if env_seed is not None:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = None

    try:
        return asyncio.run(
            _fetch_with_cache(
                company,
                client,
                model_name,
                seed,
                return_cache_info,
            )
        )
    finally:
        if needs_close:
            try:
                client.close()
            except Exception:
                pass


async def async_fetch_company_web_info(
    company: Union[str, Company],
    model: Optional[str] = None,
    *,
    seed: Optional[int] = None,
    return_cache_info: bool = False,
    client: Optional[openai.AsyncOpenAI] = None,
) -> Union[Optional[str], Tuple[Optional[str], bool]]:
    """Asynchronously fetch company info using OpenAI."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")

    needs_close = False
    if client is None:
        client = openai.AsyncOpenAI(api_key=api_key)
        needs_close = True

    model_name = model or os.getenv("OPENAI_MODEL") or "gpt-4o"

    if seed is None:
        env_seed = os.getenv("OPENAI_SEED")
        if env_seed is not None:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = None

    try:
        return await _fetch_with_cache(
            company,
            client,
            model_name,
            seed,
            return_cache_info,
        )
    finally:
        if needs_close:
            close_method = getattr(client, "aclose", None)
            if close_method is None:
                close_method = getattr(client, "close", None)
            if close_method:
                try:
                    result = close_method()
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    pass


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


def _parse_bool(value: object) -> Optional[bool]:
    """Return ``True`` or ``False`` for common boolean-like strings."""

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def parse_llm_response(response: str, *, raise_on_missing: bool = False) -> Optional[LLMOutput]:
    """Extract sanitized fields and the ``supportive`` value from the LLM response.

    Parameters
    ----------
    response:
        The raw text returned by the language model.
    raise_on_missing:
        When ``True`` and the JSON payload lacks required keys such as
        ``"supportive"`` or ``"is_business"``, a ``KeyError`` is raised. When
        ``False`` (the default), ``None`` is returned in that case.
    """

    if not response:
        logger.warning("Empty LLM response")
        return None

    match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", response, re.IGNORECASE | re.DOTALL
    )
    if not match:
        logger.warning("LLM response missing JSON block")
        return None

    json_text = match.group(1).strip()
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(json_text)
        except Exception:
            logger.warning("Failed to parse JSON from LLM response")
            return None
        if not isinstance(data, dict):
            logger.warning("LLM JSON payload not a dictionary")
            return None

    missing = [k for k in ("supportive", "is_business") if k not in data]
    if missing:
        if raise_on_missing:
            raise KeyError(
                "Missing required field(s): " + ", ".join(sorted(missing))
            )
        logger.warning("LLM output missing required field(s): %s", ", ".join(sorted(missing)))
        return None

    supportive = _parse_support_value(data.get("supportive"))

    subcat = data.get("sub_category") or data.get("subcategory")
    if isinstance(subcat, str):
        subcat = subcat.strip()
    else:
        subcat = None


    is_business = _parse_bool(data.get("is_business"))

    malformed = _parse_bool(data.get("is_possibly_malformed"))

    business_summary = (
        data.get("business_model_summary")
        or data.get("business_model")
        or data.get("summary")
    )
    if isinstance(business_summary, str):
        business_summary = business_summary.strip()
    else:
        business_summary = None

    justification = data.get("justification")
    if isinstance(justification, str):
        justification = justification.strip()
    else:
        justification = None

    try:
        result = LLMOutput(
            supportive=supportive,
            is_business=is_business,
            sub_category=subcat,
            business_model_summary=business_summary,
            justification=justification,
            is_possibly_malformed=malformed,
            raw=data,
        )
    except Exception:
        logger.warning("Failed to construct LLMOutput", exc_info=True)
        return None

    if result.supportive is None or result.is_business is None:
        logger.warning("Parsed LLM output missing critical values")

    return result


async def async_report_to_abstract(
    report: str,
    model: Optional[str] = None,
    *,
    seed: Optional[int] = None,
) -> Optional[str]:
    """Summarize a report as a scientific paper abstract using an LLM."""

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

    input_text = (
        "You transform reports into concise scientific paper abstracts.\n"
        "Rewrite the following report as a scientific paper abstract, "
        "highlighting the conclusions and referencing key statistics.\n\n"
        + report
    )

    kwargs = {"model": model_name, "input": input_text}
    if seed is not None:
        kwargs["seed"] = seed

    try:
        response = await client.responses.create(**kwargs)
    except Exception:
        logger.exception("API request failed during report summarization")
        raise

    return getattr(response, "output_text", None)


def report_to_abstract(
    report: str,
    model: Optional[str] = None,
    *,
    seed: Optional[int] = None,
) -> Optional[str]:
    """Synchronous wrapper for :func:`async_report_to_abstract`."""

    return asyncio.run(
        async_report_to_abstract(report, model, seed=seed)
    )
