from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Self, TypeVar

from pydantic import BaseModel, PrivateAttr


T = TypeVar("T")


class Messages(BaseModel):
    logs: list[dict[str, Any]]

    def recent(self, num: int=5) -> Messages:
        return Messages(self.logs[-min(num, len(self.logs)):])
    
    def append_log(self, log: Any) -> None:
        self.logs.append(log)
    
    def _append_message(self, role: str, message: str) -> None:
        self.logs.append({"role": role, "content": message})
    
    def append_user_message(self, message: str) -> None:
        self._append_message("user", message)
    
    def append_ai_message(self, message: str) -> None:
        self._append_message("assistant", message)
    
    def append_system_message(self, message: str) -> None:
        self._append_message("system", message)


class LLM(BaseModel, ABC):
    _model: str | None = PrivateAttr(default=None)

    def model(self, model: str) -> LLM:
        llm = self.model_copy()
        llm._model = model
        return llm
    
    def _model_check(self, model: str | None) -> str:
        if model:
            return model
        elif self._model:
            return self._model
        else:
            raise ValueError("model")

    @abstractmethod
    def generate(
        self,
        *,
        model: str | None = None,
        prompt: str | None = None,
        messages: str | Messages,
        strict: T | None = None
    ) -> str | T:
        pass

    @abstractmethod
    async def async_generate(
        self,
        *,
        model: str | None = None,
        prompt: str | None = None,
        messages: str | Messages,
        strict: T | None = None
    ) -> str | T:
        pass
