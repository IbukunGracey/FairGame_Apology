from typing import Dict, Any, List, Optional

from src.results_processing.agent_info import AgentInfo

class GameData:
    """
    Encapsulates all relevant data for a single game.
    """


    def __init__(
        self,
        game_id: str,
        language: Optional[str],
        n_rounds: Optional[int],
        n_rounds_is_known: bool,
        agents_communicate: bool,
        agents: List[AgentInfo],
        agents_round_data: Dict[str, Dict[str, List[Any]]],
        experiment_mode: str = "interactive",
        scenario_metadata: Dict[str, Any] = None
    ):

        self.game_id = game_id
        self.language = language
        self.n_rounds = n_rounds

        self.n_rounds_is_known = n_rounds_is_known

        self.agents_communicate = agents_communicate

        self.agents = agents

        self.agents_round_data = agents_round_data

        self.experiment_mode = experiment_mode

        self.scenario_metadata = scenario_metadata or {}

    def to_dict(self) -> Dict[str, Any]:

        row = {
            "game_id": self.game_id,

            "language": self.language,

            "experiment_mode":
                self.experiment_mode,

            "n_rounds_is_known":
                self.n_rounds_is_known,

            "max_rounds": self.n_rounds,

            "played_rounds": max(
                len(agent_data.get("strategies", []))
                for agent_data in self.agents_round_data.values()
            ) if self.agents_round_data else 0,

            "agents_communicate":
                self.agents_communicate,
        }

        if self.scenario_metadata:
            for key in ("scenario_id", "current_outcome", "memory_type",
                        "previous_outcome", "previous_apology_state"):
                if key in self.scenario_metadata:
                    row[key] = self.scenario_metadata[key]

        for idx, agent in enumerate(self.agents, start=1):

            prefix = f"agent{idx}_"

            row.update(
                agent.to_dict(prefix=prefix)
            )

            agent_name = agent.name

            round_data = self.agents_round_data.get(
                agent_name,
                {
                    "strategies": [],
                    "scores": [],
                    "messages": [],
                    "apology_decisions": [],
                    "apology_messages": [],
                    "reason_codes": [],
                    "next_actions": []
                }
            )

            row[f"{prefix}strategies"] = (
                round_data.get("strategies", [])
            )

            row[f"{prefix}scores"] = (
                round_data.get("scores", [])
            )

            row[f"{prefix}messages"] = (
                round_data.get("messages", [])
                if self.agents_communicate
                else []
            )

            row[f"{prefix}apology_decisions"] = (
                round_data.get(
                    "apology_decisions",
                    []
                )
            )

            row[f"{prefix}apology_messages"] = (
                round_data.get(
                    "apology_messages",
                    []
                )
            )

            row[f"{prefix}reason_codes"] = (
                round_data.get(
                    "reason_codes",
                    []
                )
            )

            row[f"{prefix}next_actions"] = (
                round_data.get(
                    "next_actions",
                    []
                )
            )

        return row
