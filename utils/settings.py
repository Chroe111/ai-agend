from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    agents_file: str = "data/agents.json"
    areas_file: str = "data/areas.json"

    OPENAI_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
