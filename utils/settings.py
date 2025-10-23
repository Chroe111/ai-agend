from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    agents_file: str = "data/agents.json"
    areas_file: str = "data/areas.json"

    OPENAI_API_KEY: str


class IDGenerator:
    def __init__(self, prefix: str) -> None:
        self.current_id = 0
        self.prefix = prefix
    
    def generate(self) -> str:
        id = self.current_id
        self.current_id += 1
        return f"{self.prefix}_{str(id).zfill(3)}"
