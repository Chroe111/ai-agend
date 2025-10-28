from __future__ import annotations
import asyncio
import json
import re

import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr

from .agent import AgentList
from .action import Action
from utils.llm.base import LLM
from utils.functions import cleaned, collection_search


class Area(BaseModel):
    id: str
    name: str
    description: str
    agents: list[str] = Field(default_factory=list)
    action_log: list[str] = Field(default_factory=list)
    summary: str = Field(default="")

    @property
    def name_with_id(self) -> str:
        return f"{self.name} ({self.id})"

    @property
    def info(self) -> str:
        people = "\n".join(map(lambda x: f"- {x}", self.agents))
        return cleaned(
            """
            ### {}
            {}

            ■情報
            {}

            ■ここにいる人
            {}
            """,
            self.name_with_id, self.description, self.summary, people
        )
    
    def append_action_log(self, *log: str) -> None:
        self.action_log.extend(log)
    
    async def update_info(self, llm: LLM, global_info: str) -> str:
        prompt = cleaned(
            """
            あなたはとある人々が暮らす街の管理システムです。
            エリア: {area} で起きた行動をもとに、現在のエリアの情報を更新してください。
            ### 方針
            - 1~2行程度で簡潔にまとめる。
            - イベントや事件等、大きな影響を与えうる場合は特筆する。
            - 特に書くことがない場合、いつも通りの日常が流れていることを描写する。
            """,
            area=self.name_with_id
        )
        message = cleaned(
            """
            ## グローバル情報
            {}
            ## 行動ログ
            {}
            """,
            global_info, "\n".join(self.action_log)
        )

        info = await llm.async_generate(
            prompt=prompt,
            messages=message
        )
        self.summary = info
        return info


class Location(BaseModel):
    areas: dict[str, Area]
    _loads: pd.DataFrame = PrivateAttr()

    @property
    def view(self) -> str:
        return "\n".join(area.name_with_id for area in self.areas.values())

    @classmethod
    def from_json_file(cls, filepath: str) -> Location:
        with open(filepath) as f:
            data = json.load(f)

        df = pd.DataFrame(data["distance_matrix"])
        areas = {
            (id := f"area_{str(i).zfill(2)}"): Area(
                id=id, 
                name=name,
                description=data["places_description"][name]
            ) for i, name in enumerate(data["places"])
        }
        df = df.set_axis(areas.keys(), axis="index").set_axis(areas.keys(), axis="columns")
        df = df.map(lambda x: x // 10 + int(x % 10 >= 5))

        location = cls(areas=areas)
        location._loads = df
        return location
    
    @staticmethod
    def search_id(query: str) -> str | None:
        result = re.search(r"area_\d{2}", query)
        return None if result is None else result.group()
    
    def search(self, query: str) -> Area | None:
        return self.areas.get(self.search_id(query), None)
    
    def update_log(self, action_logs: list[Action]) -> None:
        logs = {area_id: [] for area_id in self.areas.keys()}
        for action in action_logs:
            logs[action.actor.area].append(action.log(None))
        for area_id, log in logs.items():
            self.areas[area_id].action_log = log
    
    def update_agents(self, agent_list: AgentList) -> None:
        agent_changes: dict[str, list[str]] = {}
        for _, agent in agent_list:
            if agent.area in agent_changes:
                agent_changes[agent.area].append(agent.info)
            else:
                agent_changes[agent.area] = [agent.info]
        
        for area_id, agents in agent_changes.items():
            self.areas[area_id].agents = agents
    
    async def update(self, llm: LLM, action_logs: list[Action], agent_list: AgentList, global_info: str) -> None:
        self.update_agents(agent_list)
        self.update_log(action_logs)
        await asyncio.gather(*(area.update_info(llm, global_info) for area in self.areas.values()))

    def travel_time(self, departure: str, arrival: str) -> int:
        if not departure in self.areas:
            raise KeyError(departure)
        if not arrival in self.areas:
            raise KeyError(arrival)
        return int(self._loads[departure][arrival])
