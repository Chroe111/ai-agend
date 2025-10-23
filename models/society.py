from __future__ import annotations
import asyncio

from openai import AsyncOpenAI
from pydantic import BaseModel

from .action import Action, Wait, Talk, Eat, Sleep, Move, OtherAction
from .agent import Agent, AgentList
from .area import Location
from utils import settings
from utils.time import Clock
from utils.functions import cleaned


class Society(BaseModel):
    agent_list: AgentList
    location: Location
    info_text: str = ""
    clock: Clock

    @property
    def info(self) -> str:
        return cleaned(
            """
            ■現在時刻: {time},

            ■移動可能エリア
            {area}

            ■情報
            {info}
            """,
            time=self.clock.now,
            area=self.location.view,
            info=self.info_text
        )
    
    def __init__(
            self,
            agents_file: str=settings.agents_file,
            areas_file: str=settings.areas_file,
            clock: Clock | None=None) -> None:
        location = Location.from_json_file(areas_file)
        agent_list = AgentList.from_json_file(agents_file, list(location.areas.keys()))
        location.update(agent_list)
        if clock is None:
            clock = Clock()
        super().__init__(agent_list=agent_list, location=location, clock=clock)

    async def evaluate_action(
            self, 
            actor: Agent, 
            raw_action: dict[str, str], 
            client: AsyncOpenAI) -> Action:
        await asyncio.sleep(0)

        target = self.agent_list.search(raw_action.get("target", ""))
        action_type = raw_action.pop("type")
        try:
            if action_type == "talk":
                assert target
                action = Talk(actor, target, self.clock.now, raw_action["content"])
            elif action_type == "eat":
                action = Eat(actor, self.clock.now, raw_action["food"])
            elif action_type == "sleep":
                action = Sleep(actor, self.clock.now)
            elif action_type == "move":
                assert (destination := self.location.search(raw_action["destination"]))
                duration = self.location.travel_time(actor.area, destination.id)
                action = Move(actor, self.clock.now, duration, destination.id, raw_action.get("means"))
            elif action_type == "other":
                evaluated_action = await OtherAction.evaluate(
                    raw_action, 
                    actor, 
                    self.location.areas[actor.area].name_with_id,
                    client
                )
                assert evaluated_action.allow
                action = OtherAction(
                    actor, 
                    target, 
                    self.clock.now, 
                    evaluated_action.duration, 
                    evaluated_action.action
                )
            else:
                raise ValueError
        except Exception:
            action = Wait(actor, self.clock.now)

        return action
    
    async def step(self, agent: Agent, client: AsyncOpenAI) -> Action:
        raw_action = await agent.act(client, self.location.search(agent.area), self.info)
        action = await self.evaluate_action(agent, raw_action, client)

        agent.action_timer = action.duration
        agent.sleepiness += action.fatigue
        agent.hungry += action.effort
        agent.status = action.status
        agent.append_action_log(action.log(agent))

        return action
    
    async def step_all(self, client: AsyncOpenAI) -> None:
        actions = await asyncio.gather(*(self.step(agent, client) for _, agent in self.agent_list))
        for action in actions:
            for target in action.target:
                action.log(target)
        self.location.update(self.agent_list)
