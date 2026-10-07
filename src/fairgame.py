from src.payoff_matrix import PayoffMatrix
from src.game_history import GameHistory
from src.game_round import GameRound
from src.round_outcome_provider import (
    InteractiveOutcomeProvider
)
from src.history_builder import HistoryBuilder

class FairGame:

    def __init__(
        self,
        name,
        language,
        agents,
        n_rounds,
        n_rounds_known,
        payoff_matrix_data,
        game_prompt_template,
        apology_prompt_template,
        stop_conditions,
        agents_communicate,
        config,
        experiment_mode="interactive",
        round_outcome_provider=None,
        history_builder=None,
        goal_provider=None,
        controlled_prompt_template=None,
        controlled_next_action_template=None,
        scenario_metadata=None
    ):

        self.name = name

        self.language = language

        self.agents = agents

        self.n_rounds = int(n_rounds)

        self.n_rounds_known = (
            self._str2bool(n_rounds_known)
        )

        self.game_prompt_template = (
            game_prompt_template
        )

        self.apology_prompt_template = (
            apology_prompt_template
        )

        self.controlled_prompt_template = (
            controlled_prompt_template
        )

        self.controlled_next_action_template = (
            controlled_next_action_template
        )

        self.stop_conditions = stop_conditions

        self.agents_communicate = (
            self._str2bool(
                agents_communicate
            )
        )

        self.current_round = 1

        self.history = GameHistory()

        self.choices_made = []

        self.payoff_matrix = PayoffMatrix(
            payoff_matrix_data,
            language
        )

        self.config = config

        self.experiment_mode = experiment_mode

        self.apology_enabled = (
            config.get(
                "apology",
                {}
            ).get(
                "enabled",
                False
            )
        )

        self.round_outcome_provider = (
            round_outcome_provider
            or InteractiveOutcomeProvider()
        )

        self.history_builder = (
            history_builder
            or HistoryBuilder("memory_all")
        )

        self.goal_provider = goal_provider

        self.scenario_metadata = scenario_metadata or {}

    def _str2bool(self, value):

        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            return (
                value.strip().lower()
                == "true"
            )

        return False

    @property
    def description(self):

        desc = {
            "name": self.name,

            "language": self.language,

            "experiment_mode":
                self.experiment_mode,

            "agents": {
                name: agent.get_info()
                for name, agent
                in self.agents.items()
            },

            "n_rounds": self.n_rounds,

            "number_of_rounds_is_known":
                self.n_rounds_known,

            "payoff_matrix":
                self.payoff_matrix.matrix_data,

            "agents_communicate":
                self.agents_communicate,

            "apology_enabled":
                self.apology_enabled
        }

        if self.scenario_metadata:
            desc.update(self.scenario_metadata)

        return desc

    def run_round(self):

        round_runner = GameRound(self)

        # ======================
        # OUTCOME (strategy + payoff)
        # ======================

        result = self.round_outcome_provider.get_outcome(
            self.current_round,
            self.agents,
            self.payoff_matrix,
            game_round=round_runner
        )

        self.choices_made.append(
            result["round_strategies"]
        )

        # ======================
        # APOLOGY
        # ======================

        if self.apology_enabled:

            round_runner.run_apology_stage()

        # ======================
        # HISTORY
        # ======================

        round_runner.update_round_history()

    # def run_round(self):
    #     """
    #     Execute one sequential FAIRGAME round.

        
    #     Flow:
    #     1. Strategy decisions
    #     2. Score attribution
    #     3. Apology decisions
    #     """

    #     round_runner = GameRound(self)

    #     # ======================
    #     # STAGE 1: STRATEGY
    #     # ======================

    #     round_strategies = (
    #         round_runner.run_strategy_stage()
    #     )

    #     self.choices_made.append(
    #         round_strategies
    #     )

    #     # ======================
    #     # ENVIRONMENT RESPONSE
    #     # ======================

    #     self.payoff_matrix.attribute_scores(
    #         list(self.agents.values()),
    #         round_strategies
    #     )

    #     # ======================
    #     # STORE SCORES
    #     # ======================

    #     for agent in self.agents.values():

    #         self.history.update_round(
    #             self.current_round,
    #             agent.name,
    #             {
    #                 "score": agent.last_score()
    #             }
    #         )

    #     # ======================
    #     # STAGE 2: APOLOGY
    #     # ======================

    #     if self.apology_enabled:

    #         round_runner.run_apology_stage()


    def stop_condition_is_met(self):

        if self.choices_made:

            last_round_choices = (
                self.choices_made[-1]
            )

            combination = next(
                (
                    k for k, v
                    in self.payoff_matrix
                    .matrix_data[
                        'combinations'
                    ].items()

                    if v == last_round_choices
                ),
                None
            )

            if combination in self.stop_conditions:
                return True

        return False

    def run(self):
        """
        Run the game until:
        - all rounds complete
        OR
        - stop condition triggered
        """

        while (
            self.current_round <= self.n_rounds
            and not self.stop_condition_is_met()
        ):

            self.run_round()

            self.current_round += 1

        return self.history

