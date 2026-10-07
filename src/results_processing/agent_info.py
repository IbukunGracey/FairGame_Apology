from typing import Dict, Any

class AgentInfo:
    """
    Holds static information about an agent,
    such as its name, LLM service, and personality.
    """


    def __init__(
        self,
        name: str,
        llm_service: str,
        personality: str,
        opponent_prob: float
    ):
        """
        Initializes an AgentInfo instance.

        Args:
            name (str):
                Name of the agent.

            llm_service (str):
                LLM service used by the agent.

            personality (str):
                Personality descriptor for the agent.

            opponent_prob (float):
                Probability that the agent knows
                the opponent's personality.
        """

        self.name = name
        self.llm_service = llm_service
        self.personality = personality

        self.opponent_personality_probability = (
            opponent_prob
        )

    def to_dict(self, prefix: str) -> Dict[str, Any]:
        """
        Converts agent metadata into a dictionary
        suitable for DataFrame construction.

        Args:
            prefix (str):
                Prefix for dictionary keys
                (e.g. "agent1_")

        Returns:
            Dict[str, Any]:
                Dictionary representation
                of static agent metadata.
        """

        return {
            f"{prefix}name": self.name,

            f"{prefix}llm": self.llm_service,

            f"{prefix}personality": self.personality,

            f"{prefix}knows_opponent_with_prob":
                self.opponent_personality_probability
        }

