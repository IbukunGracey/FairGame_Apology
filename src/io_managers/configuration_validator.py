from src.io_managers.payoff_matrix_transformer import PayoffMatrixTransformer

class ConfigValidator:
    """
    Handles validation of top-level configuration data.
    """

    REQUIRED_KEYS = {
        "name": str,
        "gamePromptFile": str,
        "nRounds": int,
        "nRoundsIsKnown": bool,
        "payoffMatrix": dict,
        "allAgentPermutations": bool,
        "agents": dict,
        "llm": str,
        "languages": list,
        "stopGameWhen": list,
        "agentsCommunicate": bool
    }

    def validate_config_structure(self, config_data: dict) -> dict:

        # Validate top-level keys
        self._check_keys(config_data, ConfigValidator.REQUIRED_KEYS)

        # Validate payoff matrix
        try:
            PayoffMatrixTransformer.validate_payoff_matrix(
                config_data["payoffMatrix"]
            )

        except KeyError:

            config_data = PayoffMatrixTransformer.transform_payoff_input(
                config_data
            )

            PayoffMatrixTransformer.validate_payoff_matrix(
                config_data["payoffMatrix"]
            )

        # Validate agents
        if not config_data["allAgentPermutations"]:

            if not self._check_agents_configuration(
                config_data["agents"]
            ):

                raise KeyError(
                    "Configuration error: invalid agent setup."
                )

        # Validate apology configuration if enabled
        if "apology" in config_data:

            self._check_apology_configuration(
                config_data["apology"]
            )

        # Validate experiment configuration if present
        if "experiment" in config_data:

            self._check_experiment_configuration(
                config_data["experiment"],
                config_data["nRounds"]
            )

        # Validate goals configuration if present
        if "goals" in config_data:

            self._check_goals_configuration(
                config_data["goals"],
                config_data.get("experiment", {})
            )

        return config_data

    def _check_keys(
        self,
        data: dict,
        required_keys: dict
    ) -> None:

        missing_keys = []
        type_errors = []

        for key, expected_type in required_keys.items():

            if key not in data:
                missing_keys.append(key)

            elif not isinstance(data[key], expected_type):

                type_errors.append(
                    (key, type(data[key]), expected_type)
                )

        if missing_keys:

            raise KeyError(
                f"Missing keys: {', '.join(missing_keys)}"
            )

        if type_errors:

            formatted_errors = ", ".join(
                f"{key} (found: {found}, expected: {expected})"
                for key, found, expected in type_errors
            )

            raise TypeError(
                f"Type errors: {formatted_errors}"
            )

    def _check_agents_configuration(
        self,
        agents_data: dict
    ) -> bool:

        num_agents = len(
            agents_data.get("names", [])
        )

        if num_agents < 2:
            return False

        personalities = agents_data.get(
            "personalities",
            {}
        )

        for _, values in personalities.items():

            if len(values) < 2:
                return False

        all_personalities_correct = all(
            len(values) == num_agents
            for values in personalities.values()
        )

        opponent_probs = agents_data.get(
            "opponentPersonalityProb",
            []
        )

        opponent_probs_correct = (
            len(opponent_probs) == num_agents
            if opponent_probs
            else False
        )

        return (
            all_personalities_correct
            and opponent_probs_correct
        )

    def _check_apology_configuration(
        self,
        apology_data: dict
    ) -> None:

        if not isinstance(apology_data, dict):

            raise TypeError(
                "apology must be a dictionary"
            )

        required_fields = {
            "enabled": bool,
            "promptFile": str,
            "decisions": dict,
            "maxMessageLength": int,
            "outputSchema": dict
        }

        for key, expected_type in required_fields.items():

            if key not in apology_data:

                raise KeyError(
                    f"apology missing required field: {key}"
                )

            if not isinstance(
                apology_data[key],
                expected_type
            ):

                raise TypeError(
                    f"apology.{key} must be "
                    f"{expected_type.__name__}"
                )

        if apology_data["maxMessageLength"] <= 0:

            raise ValueError(
                "apology.maxMessageLength must be > 0"
            )

        decisions = apology_data["decisions"]

        if not isinstance(decisions, dict):

            raise TypeError(
                "apology.decisions must be a dict"
            )

        # Validate multilingual apology labels
        for lang, mapping in decisions.items():

            if not isinstance(mapping, dict):

                raise TypeError(
                    f"apology.decisions.{lang} must be a dict"
                )
            
            # if "yes" not in mapping or "no" not in mapping:

            #     raise KeyError(
            #         f"apology.decisions.{lang} "
            #         f"must contain 'yes' and 'no'"
            #     )

            # if not isinstance(mapping["yes"], str):

            #     raise TypeError(
            #         f"apology.decisions.{lang}.yes "
            #         f"must be a string"
            #     )

            # if not isinstance(mapping["no"], str):

            #     raise TypeError(
            #         f"apology.decisions.{lang}.no "
            #         f"must be a string"
            #     )
            if "apologyDecision1" not in mapping or "apologyDecision2" not in mapping:

                raise KeyError(
                    f"apology.decisions.{lang} "
                    f"must contain 'apologyDecision1' and 'apologyDecision2'"
                )

            if not isinstance(mapping["apologyDecision1"], str):

                raise TypeError(
                    f"apology.decisions.{lang}.apologyDecision1 "
                    f"must be a string"
                )

            if not isinstance(mapping["apologyDecision2"], str):

                raise TypeError(
                    f"apology.decisions.{lang}.apologyDecision2 "
                    f"must be a string"
                )

        # Validate apology output schema
        output_schema = apology_data["outputSchema"]

        required_output_keys = {
            "apology_decision",
            "apology_message",
            "compensation_decision",
            "compensation_amount",
            "compensation_reason"
        }

        missing_output_keys = (
            required_output_keys - output_schema.keys()
        )

        if missing_output_keys:

            raise KeyError(
                "apology.outputSchema missing keys: "
                f"{', '.join(missing_output_keys)}"
            )

        # Validate compensation configuration if present
        if "compensation" in apology_data:
            comp_cfg = apology_data["compensation"]
            if not isinstance(comp_cfg, dict):
                raise TypeError("apology.compensation must be a dictionary")

            if "enabled" in comp_cfg and not isinstance(comp_cfg["enabled"], bool):
                raise TypeError("apology.compensation.enabled must be boolean")

            if "maxCompensationReasonLength" in comp_cfg:
                max_len = comp_cfg["maxCompensationReasonLength"]
                if not isinstance(max_len, int) or max_len <= 0:
                    raise ValueError("maxCompensationReasonLength must be positive int")

    VALID_EXPERIMENT_MODES = {"interactive", "controlled"}

    VALID_HISTORY_CONDITIONS = {
        "one_shot", "no_memory",
        "memory1", "memory2", "memory_all"
    }

    VALID_OUTCOMES = {"R", "S", "T", "P"}

    VALID_APOLOGY_STATES = {"AA", "AN", "NA", "NN"}

    VALID_MEMORY_TYPES = {
        "no_memory", "one_shot", "memory1"
    }

    def _check_experiment_configuration(
        self,
        experiment_data: dict,
        n_rounds: int
    ) -> None:

        if not isinstance(experiment_data, dict):
            raise TypeError(
                "experiment must be a dictionary"
            )

        mode = experiment_data.get("mode")
        if mode not in self.VALID_EXPERIMENT_MODES:
            raise ValueError(
                f"experiment.mode must be one of "
                f"{sorted(self.VALID_EXPERIMENT_MODES)}, "
                f"got '{mode}'"
            )

        history_condition = experiment_data.get(
            "historyCondition"
        )
        if history_condition is not None:
            if history_condition not in self.VALID_HISTORY_CONDITIONS:
                raise ValueError(
                    f"experiment.historyCondition must be one of "
                    f"{sorted(self.VALID_HISTORY_CONDITIONS)}, "
                    f"got '{history_condition}'"
                )

        scenario_gen = experiment_data.get(
            "scenarioGeneration", False
        )

        if mode == "controlled" and scenario_gen:
            self._check_scenario_generation_config(
                experiment_data
            )

        elif mode == "controlled":

            if "rounds" not in experiment_data:
                raise KeyError(
                    "experiment.rounds is required "
                    "for controlled mode"
                )

            rounds = experiment_data["rounds"]

            if not isinstance(rounds, list):
                raise TypeError(
                    "experiment.rounds must be a list"
                )

            if len(rounds) != n_rounds:
                raise ValueError(
                    f"experiment.rounds has {len(rounds)} "
                    f"entries but nRounds is {n_rounds}. "
                    f"They must match."
                )

            required_round_keys = {
                "outcome", "playerStrategy",
                "opponentStrategy",
                "playerPenalty", "opponentPenalty"
            }

            for i, round_cfg in enumerate(rounds):

                missing = (
                    required_round_keys - round_cfg.keys()
                )

                if missing:
                    raise KeyError(
                        f"experiment.rounds[{i}] missing "
                        f"keys: {', '.join(sorted(missing))}"
                    )

    def _check_scenario_generation_config(
        self,
        experiment_data: dict
    ) -> None:

        memory_type = experiment_data.get("memoryType")
        if memory_type is not None:
            if memory_type not in self.VALID_MEMORY_TYPES:
                raise ValueError(
                    f"experiment.memoryType must be one of "
                    f"{sorted(self.VALID_MEMORY_TYPES)}, "
                    f"got '{memory_type}'"
                )

        outcomes = experiment_data.get("outcomes")
        if outcomes is not None:
            if not isinstance(outcomes, list):
                raise TypeError(
                    "experiment.outcomes must be a list"
                )
            invalid = set(outcomes) - self.VALID_OUTCOMES
            if invalid:
                raise ValueError(
                    f"Invalid outcome codes: {invalid}. "
                    f"Valid: {sorted(self.VALID_OUTCOMES)}"
                )

        if memory_type == "memory1":
            prev_outcomes = experiment_data.get(
                "previousOutcomes"
            )
            if prev_outcomes is not None:
                if not isinstance(prev_outcomes, list):
                    raise TypeError(
                        "experiment.previousOutcomes must be a list"
                    )
                invalid = set(prev_outcomes) - self.VALID_OUTCOMES
                if invalid:
                    raise ValueError(
                        f"Invalid previousOutcomes codes: {invalid}"
                    )

            prev_apology = experiment_data.get(
                "previousApologyStates"
            )
            if prev_apology is not None:
                if not isinstance(prev_apology, list):
                    raise TypeError(
                        "experiment.previousApologyStates must be a list"
                    )
                invalid = set(prev_apology) - self.VALID_APOLOGY_STATES
                if invalid:
                    raise ValueError(
                        f"Invalid previousApologyStates: {invalid}"
                    )

    def _check_goals_configuration(
        self,
        goals_data: dict,
        experiment_data: dict
    ) -> None:

        if not isinstance(goals_data, dict):
            raise TypeError(
                "goals must be a dictionary"
            )

        if not goals_data:
            raise ValueError(
                "goals must not be empty"
            )

        goal_condition = experiment_data.get(
            "goalCondition"
        )
        if goal_condition and goal_condition not in goals_data:
            raise ValueError(
                f"experiment.goalCondition '{goal_condition}' "
                f"not found in goals. "
                f"Available: {list(goals_data.keys())}"
            )
