from itertools import product


# Outcome codes mapped to their PD semantics (player perspective).
# R = mutual cooperation, S = sucker (player cooperates, opponent defects),
# T = temptation (player defects, opponent cooperates), P = mutual defection.
OUTCOME_SEMANTICS = {
    "R": {"player": "cooperate", "opponent": "cooperate"},
    "S": {"player": "cooperate", "opponent": "defect"},
    "T": {"player": "defect",    "opponent": "cooperate"},
    "P": {"player": "defect",    "opponent": "defect"},
}

APOLOGY_STATES = ["AA", "AN", "NA", "NN"]


class ScenarioGenerator:
    """Generates independent controlled-experiment scenarios from a payoff matrix
    and experiment configuration.

    Derives strategy keys and payoffs from the matrix rather than hard-coding them,
    so the generator works for any 2-strategy game where outcomes map to R/S/T/P.
    """

    def __init__(self, payoff_matrix_data, language):
        self.matrix_data = payoff_matrix_data
        self.language = language
        self._build_outcome_map()

    def _build_outcome_map(self):
        """Map outcome codes (R/S/T/P) to strategy keys and payoffs using the
        payoff matrix.  The mapping relies on weight magnitudes to distinguish
        cooperate vs defect strategies without hard-coding strategy names.

        Convention for penalty-framed PD (higher = worse):
          - cooperate–cooperate yields the second-lowest penalty for both → R
          - defect–defect yields the second-highest penalty for both → P
          - mixed outcomes: the defector gets the lowest penalty (0), the
            cooperator gets the highest → T (for defector) / S (for cooperator)
        """
        strategies = self.matrix_data["strategies"][self.language]
        weights = self.matrix_data["weights"]
        combinations = self.matrix_data["combinations"]
        matrix = self.matrix_data["matrix"]

        # Build lookup: (strat_key_player, strat_key_opponent) → (player_payoff, opponent_payoff)
        combo_payoffs = {}
        for combo_key, strat_keys in combinations.items():
            weight_keys = matrix[combo_key]
            payoffs = tuple(weights[wk] for wk in weight_keys)
            combo_payoffs[tuple(strat_keys)] = payoffs

        # Identify cooperate/defect strategy keys by examining symmetric outcomes.
        # In a standard PD: both-cooperate penalty < both-defect penalty.
        strat_keys = list(strategies.keys())
        if len(strat_keys) != 2:
            raise ValueError("ScenarioGenerator requires exactly 2 strategies")

        s1, s2 = strat_keys
        same_s1 = combo_payoffs.get((s1, s1), (None,))[0]
        same_s2 = combo_payoffs.get((s2, s2), (None,))[0]

        if same_s1 is None or same_s2 is None:
            raise ValueError("Payoff matrix must include both symmetric combinations")

        # The strategy pair whose symmetric penalty is lower is "cooperate"
        if same_s2 < same_s1:
            cooperate_key, defect_key = s2, s1
        else:
            cooperate_key, defect_key = s1, s2

        self._cooperate_key = cooperate_key
        self._defect_key = defect_key

        self.outcome_map = {}
        for code, semantics in OUTCOME_SEMANTICS.items():
            p_key = cooperate_key if semantics["player"] == "cooperate" else defect_key
            o_key = cooperate_key if semantics["opponent"] == "cooperate" else defect_key
            payoffs = combo_payoffs[(p_key, o_key)]
            self.outcome_map[code] = {
                "player_strategy_key": p_key,
                "opponent_strategy_key": o_key,
                "player_penalty": payoffs[0],
                "opponent_penalty": payoffs[1],
            }

    def _outcome_to_round_cfg(self, outcome_code):
        info = self.outcome_map[outcome_code]
        return {
            "outcome": outcome_code,
            "playerStrategy": info["player_strategy_key"],
            "opponentStrategy": info["opponent_strategy_key"],
            "playerPenalty": info["player_penalty"],
            "opponentPenalty": info["opponent_penalty"],
        }

    def generate(self, experiment_cfg):
        """Return a list of scenario dicts for the given experiment configuration.

        Args:
            experiment_cfg: dict with keys memoryType, outcomes,
                            and (for memory1) previousOutcomes, previousApologyStates.
        """
        memory_type = experiment_cfg.get("memoryType", "no_memory")
        outcomes = experiment_cfg.get("outcomes", ["R", "S", "T", "P"])
        self._validate_outcomes(outcomes)

        if memory_type in ("no_memory", "one_shot"):
            return self._generate_no_memory(outcomes, memory_type)

        if memory_type == "memory1":
            prev_outcomes = experiment_cfg.get("previousOutcomes", ["R", "S", "T", "P"])
            prev_apology_states = experiment_cfg.get("previousApologyStates", APOLOGY_STATES)
            self._validate_outcomes(prev_outcomes)
            self._validate_apology_states(prev_apology_states)
            return self._generate_memory1(outcomes, prev_outcomes, prev_apology_states)

        if memory_type.startswith("memory"):
            n = int(memory_type.replace("memory", ""))
            raise NotImplementedError(
                f"memory{n} scenario generation is not yet supported. "
                f"Supported: no_memory, one_shot, memory1."
            )

        raise ValueError(f"Unknown memoryType: {memory_type}")

    def _generate_no_memory(self, outcomes, memory_type):
        scenarios = []
        for outcome in outcomes:
            info = self.outcome_map[outcome]
            round_cfg = self._outcome_to_round_cfg(outcome)
            scenarios.append({
                "scenario_id": outcome,
                "current_outcome": outcome,
                "memory_type": memory_type,
                "rounds": [round_cfg],
                "n_rounds": 1,
                "history_condition": "no_memory",
                **{k: info[k] for k in (
                    "player_strategy_key", "opponent_strategy_key",
                    "player_penalty", "opponent_penalty"
                )},
            })
        return scenarios

    def _generate_memory1(self, outcomes, prev_outcomes, prev_apology_states):
        strategies = self.matrix_data["strategies"][self.language]
        scenarios = []

        for current_outcome in outcomes:
            for prev_outcome, apology_state in product(prev_outcomes, prev_apology_states):
                scenario_id = f"{current_outcome}_{prev_outcome}{apology_state}"
                current_info = self.outcome_map[current_outcome]
                prev_info = self.outcome_map[prev_outcome]

                player_apologised = apology_state[0] == "A"
                opponent_apologised = apology_state[1] == "A"

                prev_round_cfg = self._outcome_to_round_cfg(prev_outcome)

                # Synthetic previous-round history for prompt injection
                prev_history_data = {
                    "player_action": strategies[prev_info["player_strategy_key"]],
                    "opponent_action": strategies[prev_info["opponent_strategy_key"]],
                    "player_penalty": prev_info["player_penalty"],
                    "opponent_penalty": prev_info["opponent_penalty"],
                    "player_apology": "apologize" if player_apologised else "do_not_apologize",
                    "opponent_apology": "apologize" if opponent_apologised else "do_not_apologize",
                }

                current_round_cfg = self._outcome_to_round_cfg(current_outcome)

                scenarios.append({
                    "scenario_id": scenario_id,
                    "current_outcome": current_outcome,
                    "previous_outcome": prev_outcome,
                    "previous_apology_state": apology_state,
                    "memory_type": "memory1",
                    "rounds": [prev_round_cfg, current_round_cfg],
                    "n_rounds": 2,
                    "history_condition": "memory1",
                    "previous_round_data": prev_history_data,
                    **{k: current_info[k] for k in (
                        "player_strategy_key", "opponent_strategy_key",
                        "player_penalty", "opponent_penalty"
                    )},
                })

        return scenarios

    def _validate_outcomes(self, outcomes):
        invalid = set(outcomes) - set(OUTCOME_SEMANTICS.keys())
        if invalid:
            raise ValueError(f"Invalid outcome codes: {invalid}. Valid: {set(OUTCOME_SEMANTICS.keys())}")

    def _validate_apology_states(self, states):
        invalid = set(states) - set(APOLOGY_STATES)
        if invalid:
            raise ValueError(f"Invalid apology states: {invalid}. Valid: {set(APOLOGY_STATES)}")
