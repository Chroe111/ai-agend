from __future__ import annotations
from abc import ABC, abstractmethod
import json
import random
import re
from typing import Literal, override

from pydantic import BaseModel

from .agent import Agent
from utils import time as timeutils
from utils.llm.base import LLM
from utils.functions import cleaned


class Action(BaseModel, ABC):
    actor: Agent
    target: list[Agent]
    time: str
    duration: int
    status: Literal["行動可能", "行動中", "睡眠中", "移動中", "死亡"]
    fatigue: int = 1
    effort: int = 1

    @abstractmethod
    def _log_text(self, actor: str, target: str) -> str:
        pass

    def log(self, agent: Agent | None) -> str:
        actor = "あなた" if self.actor == agent else self.actor.name_with_id
        target = "あなた" if agent in self.target else "、".join(map(lambda x: x.name_with_id, self.target))

        return f"■{self.time}\n{self._log_text(actor, target)}"
    

class Dead(Action):
    @override
    def __init__(self, actor: Agent, time: str) -> None:
        super().__init__(
            actor=actor,
            target=[],
            time=time,
            duration=0,
            status="死亡",
            fatigue=0,
            effort=0
        )
    
    @override
    def _log_text(self, actor: str, target: str) -> str:
        return f"{actor}は息を引き取った。"


class Wait(Action):
    @override
    def __init__(self, actor: Agent, time: str) -> None:
        super().__init__(
            actor=actor, 
            target=[],
            time=time,
            duration=1,
            status="行動可能",
            effort=0
        )
    
    @override
    def _log_text(self, actor: str, target: str) -> str:
        return f"{actor}は何もしなかった。"


class Talk(Action):
    content: str

    @property
    def speaker(self) -> Agent:
        return self.actor

    @property
    def listener(self) -> Agent | list[Agent]:
        return self.target
    
    @override
    def __init__(self, speaker: Agent, listener: Agent | list[Agent], time: str, content: str) -> None:
        super().__init__(
            actor=speaker,
            target=[listener] if isinstance(listener, Agent) else listener,
            time=time,
            duration=1, 
            status="行動中",
            content=content.replace("「", "").replace("」", "")
        )
    
    @override
    def _log_text(self, actor: str, target: str) -> str:
        return f"{actor}は{target}に話しかけた。\n「{self.content}」"


class Eat(Action):
    food: str | list[str]

    @override
    def __init__(self, actor: Agent, time: str, food: str | list[str]) -> None:
        super().__init__(
            actor=actor,
            target=[],
            time=time,
            duration=timeutils.calc(minute=30), 
            status="行動中",
            food=food
        )
    
    @override
    def _log_text(self, actor: str, target: str) -> str:
        food = self.food if isinstance(self.food, str) else "、".join(self.food)
        return f"{actor}は{food}を食べた。"


class Sleep(Action):
    faint: bool

    @override
    def __init__(self, actor: Agent, time: str, faint: bool=False) -> None:
        super().__init__(
            target=[],
            actor=actor,
            time=time,
            duration=random.randint(timeutils.calc(hour=5), timeutils.calc(hour=8)), 
            status="睡眠中",
            faint=faint
        )
    
    @override
    def _log_text(self, actor: str, target: str) -> str:
        if self.faint:
            return f"{actor}は積み重なった疲労により気絶した。"
        else:
            return f"{actor}は眠りについた。"


class Move(Action):
    destination: str
    means: str | None

    @override
    def __init__(self, actor: Agent, time: str, duration: int, destination: str, means: str | None=None) -> None:
        super().__init__(
            actor=actor,
            target=[],
            time=time, 
            duration=duration, 
            status="移動中",
            destination=destination,
            means=means
        )
    
    @override
    def _log_text(self, actor: str, target: str) -> str:
        if self.means is None:
            return f"{actor}は{self.destination}に移動した。"
        else:
            return f"{actor}は{self.means}で{self.destination}に移動した。"


class EvaluatedAction(BaseModel):
    action: str
    actor: str
    target: str | list[str] | None
    duration: str
    allow: bool
    thinking: list[str]


class OtherAction(Action):
    action: str

    def __init__(
            self, 
            actor: Agent, 
            target: Agent | list[Agent] | None, 
            time: str, 
            duration: str, 
            action: str
    ) -> None:
        super().__init__(
            actor=actor,
            target=[] if target is None else [target] if isinstance(target, Agent) else target,
            action=action,
            time=time,
            duration=timeutils.parse(duration),
            status="行動中"
        )
    
    @override
    def _log_text(self, actor: str, target: str):
        param = {}
        if re.search(r"\{actor\}", self.action):
            param["actor"] = actor
        if re.search(r"\{target\}", self.action):
            param["target"] = target
        return self.action.format(**param)
    
    @classmethod
    async def evaluate(cls, raw_action: str, agent: Agent, area_info: str, llm: LLM) -> EvaluatedAction:
        prompt = cleaned(
            """
            あなたはAIエージェントを用いた社会シミュレーション実験の監督システムです。
            エージェントの行動を観察し、構造化して出力してください。
            以下の手順に従うこと。思考の過程も出力する。
            1. 行動を抽出する
              - 1文程度でなるべく簡潔に記述する。複合した行動は極力避ける。
              - 行動を起こしたエージェントを`{actor}`、行動の対象を`{target}`のように表記する。
              - 例: {actor}は{target}に向かってボールを投げた。
            2. 所要時間を算出する
              - iso8601形式(例: PT1H30M)で出力する。最小単位はPT10M。
              - 目安: (食事: PT30M、運動する: PT1H)
            3. 行動の**反則性**を評価する
              - 実験が破綻してしまうほどの、**シミュレーションの枠組みを超えた行動**を棄却することを目的とする。
              - 棄却すべき行動例: ロケットで宇宙へ飛ぶ、存在しないエージェント / エリアを指定するなど
              - **倫理的でない行動**とは区別する。他人の物を盗んだり、他人の命を奪っても社会シミュレーションは継続可能である。
            """, actor="{actor}", target="{target}"
        )
        message = cleaned(
            """
            ## エージェント情報
            {agent_info}
            ## エリア情報
            {area_info}
            ## エージェントの行動
            {action}
            """,
            agent_info=agent.info,
            area_info=area_info,
            action=json.dumps(raw_action)
        )
        action = await llm.async_generate(
            prompt=prompt,
            messages=message,
            schema=EvaluatedAction
        )
        return action
