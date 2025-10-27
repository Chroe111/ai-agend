from typing import TypeVar

import openai
from pydantic import PrivateAttr

from .base import Messages, LLM
from utils import settings


T = TypeVar("T")


class OpenAI(LLM):
    _client: openai.OpenAI = PrivateAttr()
    _async_client: openai.AsyncOpenAI = PrivateAttr()

    def __init__(self, api_key: str = settings.OPENAI_API_KEY) -> None:
        if not isinstance(api_key, str):
            raise TypeError(f"`api_key`: {str(api_key)}")
        
        super().__init__()
        self._client = openai.OpenAI(api_key=api_key)
        self._async_client = openai.AsyncOpenAI(api_key=api_key)

    def _create_params(
            self, 
            *,
            model: str, 
            prompt: str | None, 
            messages: str | Messages
    ) -> dict[str, str | list[dict[str, str]]]:
        params = {"model": self._model_check(model)}
        
        if prompt and isinstance(prompt, str):
            params["instructions"] = prompt
        elif not prompt is None:
            raise TypeError(f"`prompt`: {type(prompt)}")

        if isinstance(messages, str):
            params["input"] = messages
        elif isinstance(messages, Messages):
            params["input"] = messages.logs
        else:
            raise TypeError(f"`messages`: {type(messages)}")
        
        return params
    
    def generate(
            self,
            *,
            model: str | None = None,
            prompt: str | None = None,
            messages: str | Messages,
            schema: T | None = None
    ) -> str | T:
        params = self._create_params(model=model, prompt=prompt, messages=messages)
        if schema is None:
            response = self._client.responses.create(**params)
            return response.output_text
        else:
            params["text_format"] = schema
            response = self._client.responses.parse(**params)
            return response.output_parsed
    
    async def async_generate(
            self,
            *,
            model: str | None = None,
            prompt: str | None = None,
            messages: str | Messages,
            schema: T | None = None
    ) -> str | T:
        params = self._create_params(model=model, prompt=prompt, messages=messages)
        if schema is None:
            response = await self._async_client.responses.create(**params)
            return response.output_text
        else:
            params["text_format"] = schema
            response = await self._async_client.responses.parse(**params)
            return response.output_parsed
