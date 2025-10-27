from __future__ import annotations
import asyncio

import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr

from .agent import AgentList
from .action import Action
from utils.llm.base import LLM
from utils.functions import cleaned, collection_search


class Area(BaseModel):
    id: str
    name: str
    agents: list[str] = Field(default_factory=list)
    action_log: list[str] = Field(default_factory=list)
    summary: str = Field(default="")

    @property
    def name_with_id(self) -> str:
        return f"{self.name} ({self.id})"

    @property
    def info(self) -> str:
        people = "\n".join(f"- {agent}" for agent in self.agents)
        return f"{self.summary}\n### {self.name_with_id}\n■ここにいる人\n{people}"
    
    def append_action_log(self, *log: str) -> None:
        self.action_log.extend(log)
    
    async def update_info(self, llm: LLM, global_info: str) -> str:
        prompt = cleaned(
            """
            あなたはとある人々が暮らす街の管理システムです。
            エリア: {area} で起きた行動をもとに、現在のエリアの情報を更新してください。
            ### 方針
            - 1~2行程度で簡潔にまとめる。
            - 個々の行動については触れず、サマリーとして記述する。
              - ただしイベントや事件等、大きな影響を与えうる場合はその限りではない。
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
        df = pd.read_json(filepath)
        areas = {(id := f"area_{str(i).zfill(2)}"): Area(id=id, name=name) for i, name in enumerate(df.index)}
        df = df.set_axis(areas.keys(), axis="index").set_axis(areas.keys(), axis="columns")
        df = df.map(lambda x: x // 10 + int(x % 10 >= 5))
        location = cls(areas=areas)
        location._loads = df
        return location
    
    def search(self, query: str) -> Area:
        return collection_search(self.areas, r"area_\d{2}", query)
    
    def update_log(self, action_logs: list[Action]) -> None:
        print(action_logs)
        for action in action_logs:
            self.areas[action.actor.area].append_action_log(action.log(None))
    
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
