from retry import retry
from src.prompt_creator import PromptCreator

class GameRound:
    """
    Represents a single round in the FAIRGAME system.

    Supports two-stage execution:
    - Stage 1: strategy selection
    - Stage 2: apology decision (post-outcome)
    """

    def __init__(self, game):
        self.game = game
        self.round_number = game.current_round

    # ============================
    # STAGE 1: STRATEGY SELECTION
    # ============================

    def run_strategy_stage(self):
        """
        Execute only strategy selection for all agents.
        """

        round_strategies = []

        for agent in self.game.agents.values():

            prompt = self.create_prompt(
                agent,
                phase="choose"
            )

            strategy = self._execute_agent_strategy(
                agent,
                prompt
            )

            round_strategies.append(strategy["strategy"])

        return round_strategies

    # ============================
    # STAGE 2: APOLOGY DECISION
    # ============================

    def run_apology_stage(self):

        apology_phase = (
            "controlled"
            if self.game.experiment_mode == "controlled"
            else "apology"
        )

        for agent in self.game.agents.values():

            prompt = self.create_prompt(
                agent,
                phase=apology_phase
            )

            response = agent.execute_apology_round(
                prompt
            )

            apology_decision = response.get("apology_decision", "NO")
            apology_message = response.get("apology_message", "")
            reason_code = response.get("reason_code", "")

            agent.add_apology(
                apology_decision,
                apology_message,
                reason_code
            )

            print(f"Apology Decision: {apology_decision}")
            print(f"Apology Message: {apology_message}")
            if reason_code:
                print(f"Reason Code: {reason_code}")
            print()

            self.game.history.update_round(
                self.round_number,
                agent.name,
                {
                    "apology_decision": apology_decision,
                    "apology_message": apology_message,
                    "reason_code": reason_code
                }
            )

        # Controlled experiments: separate next-action prompt
        if self.game.experiment_mode == "controlled":
            self._run_next_action_stage()

    def _run_next_action_stage(self):
        for agent in self.game.agents.values():
            prompt = self.create_prompt(
                agent,
                phase="controlled_next_action"
            )
            next_action = self._execute_next_action(
                agent, prompt
            )
            self.game.history.update_round(
                self.round_number,
                agent.name,
                {"next_action": next_action}
            )

    # ============================
    # HISTORY UPDATE
    # ============================

    def update_round_history(self):
        """
        Persist each agent's strategy and score for this round into the
        shared game history.

        Apology data (decision/message) is written separately during
        ``run_apology_stage``; ``GameHistory.update_round`` merges the two.
        """

        for agent in self.game.agents.values():

            self.game.history.update_round(
                self.round_number,
                agent.name,
                {
                    "strategy": agent.last_strategy(),
                    "score": agent.last_score()
                }
            )

    # ============================
    # PROMPT CREATION
    # ============================

    def create_prompt(self, agent, phase):

        opponents = self._get_opponents(agent)

        if phase == "controlled":
            template = self.game.controlled_prompt_template
        elif phase == "controlled_next_action":
            template = self.game.controlled_next_action_template
        elif phase == "choose":
            template = self.game.game_prompt_template
        else:
            template = self.game.apology_prompt_template

        filtered_history = self.game.history_builder.build(
            self.game.history.rounds,
            self.round_number,
            agent.name
        )

        prompt_creator = PromptCreator(
            self.game.language,
            template,
            self.game.n_rounds,
            self.game.n_rounds_known,
            self.game.payoff_matrix,
            self.game.config,
            goal_provider=self.game.goal_provider
        )

        return prompt_creator.fill_template(
            agent,
            opponents,
            self.round_number,
            filtered_history,
            phase
        )

    def _get_opponents(self, agent):
        return [
            a for a in self.game.agents.values()
            if a != agent
        ]

    # ============================
    # STRATEGY EXECUTION
    # ============================

    @retry(tries=10, delay=1)
    def _execute_agent_strategy(self, agent, prompt):
        """
        Executes Stage 1 LLM call.
        """

        strategy = agent.execute_strategy_round(prompt)

        # validate strategy
        strategy_key = next(
            (
                key for key, val
                in self.game.payoff_matrix.strategies.items()
                if val.lower() in strategy.lower()
            ),
            None
        )

        if not strategy_key:
            raise ValueError(
                f"Invalid strategy: {strategy}"
            )

        # store in agent memory
        choice = self.game.payoff_matrix.strategies[strategy_key]

        agent.add_strategy(choice)

        # Report the agent's resolved choice to the console.
        print(f"Game Choice: {choice}\n")

        return {
            "strategy": strategy_key
        }

    @retry(tries=10, delay=1)
    def _execute_next_action(self, agent, prompt):
        """
        Controlled experiment: separate LLM call for
        next-round strategy choice.
        """

        response = agent.execute_strategy_round(prompt)

        strategy_key = next(
            (
                key for key, val
                in self.game.payoff_matrix.strategies.items()
                if val.lower() in response.lower()
            ),
            None
        )

        if not strategy_key:
            raise ValueError(
                f"Invalid next action: {response}"
            )

        choice = self.game.payoff_matrix.strategies[strategy_key]
        agent.add_next_action(choice)

        print(f"Next Action: {choice}\n")

        return choice

