import json
import re
from typing import Any, Dict, List

from src.llm_connectors.llm_factory_connector import execute_prompt


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Parse a JSON object from an LLM response, tolerating common wrappers
    such as ```json fenced blocks or surrounding prose.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """

    cleaned = text.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```).
    fence_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE
    )
    if fence_match:
        cleaned = fence_match.group(1)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fall back to the first {...} block found anywhere in the text.
    brace_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(0))

    raise ValueError("No JSON object found in response.")

class Agent:
    """
    Represents an agent interacting with an LLM service.


    The agent stores:
    - strategies
    - scores
    - apology decisions
    - apology messages
    """

    def __init__(self,name: str, llm_service: str,personality: str, opponent_personality_prob: int) -> None:

        self.name: str = name
        self.llm_service: str = llm_service
        self.personality: str = personality
        self.opponent_personality_prob: int = (
            opponent_personality_prob
        )

        # Round-level gameplay history
        self.strategies: List[str] = []
        self.scores: List[int] = []

        # Round-level apology history
        self.apology_decisions: List[str] = []
        self.apology_messages: List[str] = []
        self.reason_codes: List[str] = []
        self.next_actions: List[str] = []

        # Compensation history
        self.compensation_decisions: List[str] = []
        self.compensation_amounts: List[float] = []
        self.compensation_reasons: List[str] = []
        self.adjusted_scores: List[float] = []  # Keep as float to avoid rounding compensation

    def execute_strategy_round(self, prompt: str) -> str:
        """
        Executes Stage 1:
        strategy selection.

        Expected LLM output:
            OptionA
            or
            OptionB

        Returns:
            str:
                Selected strategy.
        """

        response = execute_prompt(
            self.llm_service,
            prompt
        )

        strategy = response.strip()

        return strategy

    def execute_apology_round(
        self,
        prompt: str
    ) -> Dict[str, str]:
        """
        Executes Stage 2:
        apology decision.

        Expected JSON output:

        {
            "apology_decision": "YES",
            "apology_message": "Sorry."
        }

        Returns:
            Dict[str, str]:
                Parsed apology response.
        """

        response = execute_prompt(
            self.llm_service,
            prompt
        )

        try:

            parsed_response = _extract_json_object(response)

        except (json.JSONDecodeError, ValueError) as e:

            raise ValueError(
                f"Agent '{self.name}' "
                f"returned invalid apology JSON:\n"
                f"{response}"
            ) from e

        return {
            "apology_decision":
                parsed_response.get(
                    "apology_decision",
                    "NO"
                ),

            "apology_message":
                parsed_response.get(
                    "apology_message",
                    ""
                ),

            "reason_code":
                parsed_response.get(
                    "reason_code",
                    ""
                ),

            "compensation_decision":
                parsed_response.get(
                    "compensation_decision",
                    "NO"
                ),

            "compensation_amount":
                parsed_response.get(
                    "compensation_amount",
                    0
                ),

            "compensation_reason":
                parsed_response.get(
                    "compensation_reason",
                    ""
                ),

            "next_action":
                parsed_response.get(
                    "next_action",
                    ""
                )
        }

    def add_strategy(self, strategy: str) -> None:
        """
        Record a strategy choice.
        """
        self.strategies.append(strategy)

    def last_strategy(self) -> str:
        """
        Retrieve most recent strategy.
        """
        return self.strategies[-1]

    def add_score(self, score: int) -> None:
        """
        Record a score.
        """
        self.scores.append(score)

    def last_score(self) -> int:
        """
        Retrieve most recent score.
        """
        return self.scores[-1]

    def add_apology(
        self,
        apology_decision: str,
        apology_message: str,
        reason_code: str = ""
    ) -> None:

        self.apology_decisions.append(
            apology_decision
        )

        self.apology_messages.append(
            apology_message
        )

        self.reason_codes.append(reason_code)

    def add_next_action(self, next_action: str) -> None:
        self.next_actions.append(next_action)

    def last_next_action(self) -> str:
        return self.next_actions[-1] if self.next_actions else ""

    def last_apology_decision(self) -> str:
        """
        Retrieve most recent apology decision.
        """
        return self.apology_decisions[-1]

    def last_apology_message(self) -> str:
        """
        Retrieve most recent apology message.
        """
        return self.apology_messages[-1]

    def last_reason_code(self) -> str:
        return self.reason_codes[-1] if self.reason_codes else ""

    def add_compensation(self, decision: str, amount: float, reason: str) -> None:
        """Record compensation offered this round."""
        self.compensation_decisions.append(decision)
        self.compensation_amounts.append(amount)
        self.compensation_reasons.append(reason)

    def last_compensation_decision(self) -> str:
        return self.compensation_decisions[-1] if self.compensation_decisions else "NO"

    def last_compensation_amount(self) -> float:
        return self.compensation_amounts[-1] if self.compensation_amounts else 0.0

    def last_compensation_reason(self) -> str:
        return self.compensation_reasons[-1] if self.compensation_reasons else ""

    def add_adjusted_score(self, score: float) -> None:
        """Record adjusted penalty after compensation."""
        self.adjusted_scores.append(score)

    def last_adjusted_score(self) -> float:
        """Return adjusted penalty, or original if not yet adjusted."""
        return self.adjusted_scores[-1] if self.adjusted_scores else float(self.last_score())

    def get_info(self) -> Dict[str, Any]:
        """
        Retrieve static metadata.
        """

        return {
            "name": self.name,

            "llm_service":
                self.llm_service,

            "personality":
                self.personality,

            "opponent_personality_probability":
                self.opponent_personality_prob
        }
