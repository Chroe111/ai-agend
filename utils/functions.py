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


class Clock:
    @staticmethod
    def check(day: int, hour: int, minute: int) -> bool:
        if not isinstance(day, int):
            return False
        if day < 0:
            return False
        if not isinstance(hour, int):
            return False
        if hour < 0 or hour >= 24:
            return False
        if not isinstance(minute, int):
            return False
        if minute < 0 or minute >= 60:
            return False
        return True
    
    @classmethod
    def calc(cls, day: int, hour: int, minute: int) -> int:
        if not cls.check(day, hour, minute):
            raise ValueError
        return day * 24 * 6 + hour * 6 + minute // 10
    
    @classmethod
    def parse(cls, text: str) -> int:
        pattern = r"P(Y(?P<day>\d+))?(H(?P<hour>\d+))?(M(?P<minute>\d+))?"
        return cls.calc(**re.search(pattern, text).groupdict())