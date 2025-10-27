from __future__ import annotations
import asyncio

from pydantic import BaseModel

from .action import Action, Dead, Wait, Talk, Eat, Sleep, Move, OtherAction
from .agent import Agent, AgentList
from .area import Location
from utils import settings
from utils.functions import cleaned
from utils.llm.base import LLM
from utils.time import Clock


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
            clock: Clock | None=None
    ) -> None:
        location = Location.from_json_file(areas_file)
        agent_list = AgentList.from_json_file(agents_file, list(location.areas.keys()))
        location.update_agents(agent_list)
        if clock is None:
            clock = Clock()
        super().__init__(agent_list=agent_list, location=location, clock=clock)

    async def evaluate_action(
            self, 
            actor: Agent, 
            raw_action: dict[str, str], 
            llm: LLM
    ) -> Action:
        await asyncio.sleep(0)

        target = [self.agent_list.search(t) for t in raw_action.get("target", "")]
        action_type = raw_action.pop("type")
        try:
            if action_type == "dead":
                action = Dead(actor, self.clock.now)
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
                action = Move(actor, self.clock.now, duration, destination.name_with_id, raw_action.get("means"))
            elif action_type == "other":
                evaluated_action = await OtherAction.evaluate(
                    raw_action, 
                    actor, 
                    self.location.areas[actor.area].name_with_id,
                    llm
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
    
    async def step(self, agent: Agent, llm: LLM) -> Action | None:
        if agent.status == "行動可能":
            raw_action = await agent.act(llm, self.location.search(agent.area).info, self.info)
            action = await self.evaluate_action(agent, raw_action, llm)
            if action is None:
                agent.status = "死亡"
                action
            else:
                agent.action_timer = action.duration
                if isinstance(action, Move):
                    agent.moving_to = action.destination
                agent.status = action.status
                agent.append_action_log(action.log(agent))
        else:
            action = None
        
        agent.action_timer += -1
        agent.sleepiness += 1
        agent.hungry += 1
        if agent.action_timer == 0:
            if agent.status == "移動中":
                agent.area = agent.moving_to
                agent.moving_to = None
            agent.status = "行動可能"

        return action
    
    async def step_all(self, llm: LLM) -> list[Action]:
        actions = await asyncio.gather(*(self.step(agent, llm) for _, agent in self.agent_list if agent.status != "死亡"))
        self.agent_list.send_action_logs([action for action in actions if not action is None])
        await self.location.update(llm, actions, self.agent_list, self.info)
        self.clock.step()

        return actions
