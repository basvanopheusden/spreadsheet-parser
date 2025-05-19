import os
from typing import Optional

import openai


def fetch_company_web_info(company_name: str, model: Optional[str] = None) -> Optional[str]:
    """Ask an LLM to search the web for company information.

    This is a stub implementation that assumes the underlying model has the
    ability to search the web.  It sends a simple prompt to the OpenAI API
    using the key found in the ``OPENAI_API_KEY`` environment variable.  By
    default it targets a modern GPT-4 based model but callers can override the
    model either via the ``model`` parameter or the ``OPENAI_MODEL``
    environment variable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")

    openai.api_key = api_key

    model_name = model or os.getenv("OPENAI_MODEL") or "gpt-4o"

    prompt = (
        f"Search the web for information about {company_name}. "
        "Summarize the company's business model, data strategy, and likely "
        "stance on interoperability and access legislation." 
    )

    response = openai.ChatCompletion.create(
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

