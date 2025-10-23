from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

from .agent import AgentList
from utils.functions import collection_search


class Area(BaseModel):
    id: str
    name: str
    agents: list[str] = Field(default_factory=list)
    action_log: list[str] = Field(default_factory=list)

    @property
    def name_with_id(self) -> str:
        return f"{self.name} ({self.id})"

    @property
    def info(self) -> str:
        people = "\n".join(f"- {agent}" for agent in self.agents)
        return f"### {self.name_with_id}\n■ここにいる人\n{people}"


class Location(BaseModel):
    areas: dict[str, Area]
    _loads: pd.DataFrame

    @property
    def view(self) -> str:
        return "\n".join(area.name_with_id for area in self.areas.values())

    @classmethod
    def from_json_file(cls, filepath: str) -> Location:
        df = pd.read_json(filepath)
        areas = {(id := f"area_{str(i).zfill(2)}"): Area(id=id, name=name) for i, name in enumerate(df.index)}
        df.set_axis(areas.keys(), axis="index")
        df.set_axis(areas.keys(), axis="columns")
        df = df.map(lambda x: (x // 10 + int(x % 10 >= 5)) * 10)
        return cls(areas=areas, _loads=df)
    
    def search(self, query: str) -> Area:
        return collection_search(self.areas, r"area_\d{2}", query)
    
    def update(self, agent_list: AgentList) -> None:
        result: dict[str, list[str]] = {}
        for id, agent in agent_list:
            if agent.area in result:
                result[agent.area].append(id)
            else:
                result[agent.area] = [id]
        
        for area_id, agents in result.items():
            self.areas[area_id].agents = agents

    def travel_time(self, departure: str, arrival: str) -> int:
        if not departure in self.areas:
            raise KeyError(departure)
        if not arrival in self.areas:
            raise KeyError(arrival)
        return self._loads[departure][arrival]
