import re
from typing import override

from pydantic import BaseModel


class Clock(BaseModel):
    _clock: int = 0
    
    @property
    def evaluate(self) -> tuple[int, int, int]:
        return evaluate(self._clock)
    
    @property
    def now(self) -> str:
        day, hour, minute = evaluate(self._clock)
        return f"{day + 1}日目 {str(hour).zfill(2)}時{str(minute).zfill(2)}分"
    
    @override
    def __init__(self, day: int=0, hour: int=7, minute: int=0) -> None:
        self._clock = calc(day=day, hour=hour, minute=minute)

    def step(self, *, day: int=0, hour: int=0, minute: int=10) -> None:
        self._clock += calc(day=day, hour=hour, minute=minute)


def _check(day: int, hour: int, minute: int) -> bool:
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


def calc(*, day: int=0, hour: int=0, minute: int=0) -> int:
    if not _check(day, hour, minute):
        raise ValueError
    return day * 24 * 6 + hour * 6 + minute // 10


def parse(text: str) -> int:
    pattern = r"P(Y(?P<day>\d+))?(H(?P<hour>\d+))?(M(?P<minute>\d+))?"
    return calc(**re.search(pattern, text).groupdict())


def evaluate(clock: int) -> tuple[int, int, int]:
    minute = (clock % 6) * 10
    hour = (clock // 6) % 24
    day = clock // (6 * 24)
    return day, hour, minute
