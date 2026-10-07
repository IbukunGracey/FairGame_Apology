import copy


class HistoryBuilder:

    VALID_CONDITIONS = {
        "one_shot",
        "no_memory",
        "memory1",
        "memory2",
        "memory_all"
    }

    def __init__(self, condition="memory_all"):
        if condition not in self.VALID_CONDITIONS:
            raise ValueError(
                f"Invalid history condition: '{condition}'. "
                f"Must be one of {sorted(self.VALID_CONDITIONS)}"
            )
        self.condition = condition

    def build(self, full_history, current_round, agent_name=None):
        if self.condition in ("one_shot", "no_memory"):
            return {}

        if self.condition == "memory_all":
            return full_history

        n = int(self.condition.replace("memory", ""))
        return self._last_n_rounds_with_apology(
            full_history, current_round, n
        )

    def _last_n_rounds_with_apology(self, history, current_round, n):
        start_round = max(1, current_round - n)

        filtered = {}
        for round_key, agents_data in history.items():
            round_num = int(round_key.split("_")[1])

            if round_num >= start_round and round_num < current_round:
                filtered[round_key] = copy.deepcopy(
                    agents_data
                )

        return filtered
