from typing import TypeVar

from google.genai import Client
from pydantic import PrivateAttr, model_validator

from .base import LLM, Messages
from utils import settings


T = TypeVar("T")


class Gemini(LLM):
    _client: Client = PrivateAttr()

    def __init__(self, api_key: str = settings.GEMINI_API_KEY) -> None:
        if not isinstance(api_key, str):
            raise TypeError(f"`api_key`: {str(api_key)}")
        
        super().__init__()
        self._client = Client(api_key=api_key)
    
    def _create_params(
            self, 
            model: str | None, 
            prompt: str | None, 
            messages: str | Messages
    ) -> dict[str, str | list[dict[str, str]]]:
        params = {"model": self._model_check(model), "config": {}}
        
        if isinstance(prompt, str):
            params["config"]["system_instruction"] = prompt
        elif not prompt is None:
            raise TypeError(f"`prompt`: {type(prompt)}")

        if isinstance(messages, str):
            params["contents"] = messages
        elif isinstance(messages, Messages):
            params["contents"] = messages.logs
        else:
            raise TypeError(f"`messages`: {type(messages)}")
        
        return params
    
    def generate(
            self, 
            *,
            model: str | None = None,
            prompt: str | None = None, 
            messages: str | Messages, 
            schema: type | None = None
    ) -> str | T:
        
        params = self._create_params(model, prompt, messages)
        if schema is None:
            response = self._client.models.generate_content(**params)
            return response.text
        elif isinstance(schema, type):
            params["config"]["response_mime_type"] = "application/json"
            params["config"]["response_schema"] = schema
            response = self._client.models.generate_content(**params)
            return response.parsed
    
    async def async_generate(
            self, 
            *,
            model: str | None = None,
            prompt: str | None = None, 
            messages: str | Messages, 
            schema: T | None = None
    ) -> str | T:
        params = self._create_params(model, prompt, messages)
        if schema is None:
            response = await self._client.aio.models.generate_content(**params)
            return response.text
        elif isinstance(schema, type):
            params["config"]["response_mime_type"] = "application/json"
            params["config"]["response_schema"] = schema
            response = await self._client.aio.models.generate_content(**params)
            return response.parsed
        