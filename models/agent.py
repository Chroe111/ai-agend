from __future__ import annotations
from collections.abc import Iterator
import random
import re
import textwrap
from typing import Any, Literal, TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel, Field

from utils import time
from utils.llm.base import LLM
from utils.functions import cleaned, collection_search, parse_json

if TYPE_CHECKING:
    from .action import Action


class Agent(BaseModel):
    id: str
    name: str
    job: str
    character: str
    initiative: str
    sociability: str
    area: str
    moving_to: str | None = None
    sleepiness: int = 0
    hungry: int = 0
    status: Literal["行動可能", "行動中", "睡眠中", "移動中", "死亡"] = "行動可能"
    action_timer: int = 0
    thinking: str = ""
    action_log: list[str] = Field(default_factory=list)

    @property
    def name_with_id(self) -> str:
        return f"{self.name} ({self.id})"

    @property
    def info(self) -> str:
        return f"[{self.job}] {self.name_with_id}: {self.status}"

    @property
    def persona(self) -> str:
        data = []

        data.append(cleaned(
            """
            ### 基本情報
            名前: {}
            職業: {}
            性格: {}
            行動力: {}
            コミュニケーション能力: {}
            """,
            self.name_with_id, self.job, self.character, self.initiative, self.sociability
        ))

        data.append(f"### 眠気: {int(100 * self.sleepiness / time.calc(hour=20))}/100")
        if self.sleepiness < time.calc(hour=1):
            data.append("睡眠から目覚めた。")
        elif self.sleepiness < time.calc(10):
            data.append("眠気は感じない。")
        elif self.sleepiness < time.calc(hour=15):
            data.append("疲労がたまり、眠くなってきた。")
        else:
            data.append("体が限界を迎えている。そろそろ眠りに落ちてしまいそうだ。")

        data.append(f"\n### 空腹度: {int(100 * self.hungry / time.calc(hour=20))}/100")
        if self.hungry < time.calc(hour=6):
            data.append("前回の食事により腹は満たされている。")
        elif self.hungry < time.calc(day=1):
            data.append("おなかがすいてきた。")
        elif self.hungry < time.calc(day=2):
            data.append("長い間何も食べていない。行動や健康に悪影響が出始める。")
        elif self.hungry < time.calc(day=2, hour=12):
            data.append("2日近く何も食べていない。身体が衰弱し始めている。")
        else:
            data.append("体が動かない。まもなく餓死することを悟る。")
        
        if self.thinking:
            data.append(f"\n### 直前の行動の思考\n{self.thinking}")

        data.append(f"\n### 行動ログ")
        if len(self.action_log) == 0:
            data.append("情報なし")
        else:
            data.extend(self.action_log)

        return "\n".join(data)
    
    def recent_action(self, n: int=5) -> list[str]:
        return self.action_log[-(min(n, len(self.action_log))):]
    
    async def act(self, llm: LLM, area_info: str, global_info: str) -> dict[str, Any]:
        if self.hungry >= time.calc(day=3):
            return {"type": "dead"}
        
        if self.sleepiness >= time.calc(day=1):
            return {"type": "sleep", "faint": True}

        schema = textwrap.dedent(
            """
            {
                "thinking": (思考や行動指針),
                "action": {
                    "type": (行動),
                    ...
                }
            }
            """
        )
        prompt = cleaned(
            """
            あなたはとある街で日常生活を送っています。
            以下の情報を総合的に参照し、あなたの思考や行動指針を整理してください。
            それを踏まえて、あなたの次の行動を1つ宣言してください。
            選択できる行動は 5 種類あり、以下の項目を含める。
            - 他エージェントに話しかける (`"type": "talk"`)
              - `target`: 対象のエージェント ID (複数可)
              - `content`: 話しかける内容・セリフ
            - 食事をとる (`"type": "eat"`)
              - `food`: 食べるもの / 飲むもの
            - 睡眠をとる ("type": "sleep"`)
              - 引数なし
            - 他エリアに移動する (`"type": "move"`)
              - `destination`: 移動先のエリア ID
              - `means`: 移動方法 (Optional)
            - その他の行動 (`"type": "other"`)
              - `target`: 行動の対象ID (Optional)
              - `detail`: 行動内容。なるべく詳細に  
            **スキーマ**
            ```json
            {}
            ```
            """,
            schema
        )
        message = cleaned(
            """
            ## グローバル情報
            {}
            ## 周囲の状況
            {}
            ## あなたの情報
            {}
            """, 
            global_info, area_info, self.persona
        )
        
        raw_behavior = await llm.async_generate(
            prompt=prompt,
            messages=message,
        )

        behavior = parse_json(raw_behavior)
        self.thinking = behavior["thinking"]
        return behavior["action"]
    
    def append_action_log(self, *log: str) -> None:
        self.action_log.extend(log)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, value: Agent) -> bool:
        if not isinstance(value, Agent):
            return False
        return hash(self) == hash(value)


class AgentList(BaseModel):
    agents: dict[str, Agent]

    @property
    def view(self) -> dict[str, list[str]]:
        return self.agents

    @classmethod
    def from_json_file(cls, filepath: str, areas: list[str]) -> AgentList:
        df = pd.read_json(filepath)
        return cls(agents={
            (id := f"agent_{str(i).zfill(3)}"): Agent(
                id=id,
                name=row.loc["名前"],
                job=row.loc["職業"],
                character=row.loc["特徴語"],
                initiative=row.loc["行動力"],
                sociability=row.loc["コミュニケーション能力"],
                area=random.choice(areas),
                hungry=time.calc(hour=7)
            ) for i, row in df.iterrows()
        })
    
    @staticmethod
    def search_id(query: str) -> str | None:
        result = re.search(r"agent_\d{3}", query)
        return None if result is None else result.group()
    
    def search(self, query: str) -> Agent | None:
        return self.agents.get(self.search_id(query), None)

    def search(self, query: str) -> Agent | None:
        return collection_search(self.agents, r"agent_\d{3}", query)
    
    def send_action_logs(self, action_list: list[Action]) -> None:
        for action in action_list:
            for target in action.target:
                self.agents[target.id].append_action_log(action.log(target))
    
    def __iter__(self) -> Iterator[tuple[str, Agent]]:
        return iter(self.agents.items())
    