import json
import re
import textwrap
from typing import Any, TypeVar


T = TypeVar("P")


def cleaned(text: str, *args: str, **kwargs: str) -> str:
    return textwrap.dedent(text).strip().format(*args, **kwargs)


def parse_json(text: str) -> dict[str, Any] | list[dict[str, Any]]:
    result = re.search(r"(\{.*\})", re.sub(r'[\s]+', '', text))
    return None if result is None else json.loads(result.group())


def collection_search(collection: dict[str, T], pattern: str, query: str) -> T | None:
    try:
        return collection[re.search(pattern, query).group()]
    except:
        return None
