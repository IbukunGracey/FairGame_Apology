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

        max_penalty = max(self.game.payoff_matrix.weights.values())

        # Collect apology decisions from both agents
        agents_list = list(self.game.agents.values())
        for i, agent in enumerate(agents_list):
            # Get opponent to check their penalty for validation
            opponent = agents_list[1 - i] if len(agents_list) == 2 else None

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

            # Extract raw compensation values
            raw_compensation_decision = response.get("compensation_decision", "NO")
            raw_compensation_amount = response.get("compensation_amount", 0)
            raw_compensation_reason = response.get("compensation_reason", "")

            # Validate and sanitize compensation based on apology decision
            compensation_decision, compensation_amount, compensation_reason = self._validate_and_sanitize_compensation(
                apology_decision,
                raw_compensation_decision,
                raw_compensation_amount,
                raw_compensation_reason,
                max_penalty,
                opponent.last_score() if opponent else 0
            )

            # Truncate compensation reason
            comp_cfg = self.game.config.get("apology", {}).get("compensation", {})
            max_reason_len = comp_cfg.get("maxCompensationReasonLength", 30)
            compensation_reason = " ".join(compensation_reason.split()[:max_reason_len])

            # Store in agent memory
            agent.add_apology(
                apology_decision,
                apology_message,
                reason_code
            )
            agent.add_compensation(compensation_decision, compensation_amount, compensation_reason)

            # Store in history
            # When no apology, use None for compensation fields to indicate "not applicable"
            apology_given = apology_decision.upper() in ("YES", "APOLOGIZE")
            self.game.history.update_round(
                self.round_number,
                agent.name,
                {
                    "apology_decision": apology_decision,
                    "apology_message": apology_message,
                    "reason_code": reason_code,
                    "compensation_decision": compensation_decision if apology_given else None,
                    "compensation_amount": compensation_amount if apology_given else None,
                    "compensation_reason": compensation_reason if apology_given else None
                }
            )

            print(f"Apology Decision: {apology_decision}")

            if apology_decision.upper() in ("YES", "APOLOGIZE"):
                print(f"Apology Message: {apology_message}")
                if reason_code:
                    print(f"Reason Code: {reason_code}")
                print(f"Compensation Decision: {compensation_decision}")
                print(f"Compensation Amount: {compensation_amount}")
                if compensation_reason:
                    print(f"Compensation Reason: {compensation_reason}")
            else:
                if reason_code:
                    print(f"Reason Code: {reason_code}")

            print()

        # Apply compensation transfers AFTER both agents decide
        self._apply_compensation_transfers()

        # Update history with adjusted scores
        for agent in self.game.agents.values():
            self.game.history.update_round(
                self.round_number,
                agent.name,
                {"adjusted_score": agent.last_adjusted_score()}
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
    # COMPENSATION LOGIC
    # ============================

    def _validate_and_sanitize_compensation(
        self,
        apology_decision: str,
        compensation_decision: str,
        compensation_amount,
        compensation_reason: str,
        max_penalty: float,
        opponent_penalty: float
    ) -> tuple:
        """
        Validate and sanitize compensation based on the three-outcome model.

        Three valid outcomes:
        1. No apology: compensation_decision="NO", amount=0, reason=""
        2. Apology without compensation: apology_decision="YES", compensation_decision="NO", amount=0, reason provided
        3. Apology with compensation: apology_decision="YES", compensation_decision="YES", amount>0, reason provided

        Rules:
        - If no apology: force compensation_decision="NO", amount=0, reason=""
        - If apology but compensation_decision="NO": force amount=0, keep reason
        - If apology and compensation_decision="YES": validate amount>0, <=maxPenalty, <=opponent_penalty
        - Amount must not exceed opponent's current penalty (prevents negative adjusted penalty)

        Returns: (validated_decision, validated_amount, validated_reason)
        """
        # Normalize apology decision
        apology_yes = apology_decision.upper() in ("YES", "APOLOGIZE")

        # Outcome 1: No apology - no compensation allowed
        if not apology_yes:
            if compensation_decision.upper() in ("YES", "APOLOGIZE") or compensation_amount or compensation_reason:
                print(f"Warning: No apology but compensation offered. Forcing compensation_decision=NO, amount=0, reason=''")
            return "NO", 0.0, ""

        # Outcomes 2 & 3: Apology given - compensation is optional
        comp_yes = compensation_decision.upper() in ("YES", "APOLOGIZE")

        # Parse amount
        try:
            amount = float(compensation_amount) if compensation_amount not in (None, "", []) else 0.0
        except (TypeError, ValueError):
            print(f"Warning: Invalid compensation_amount '{compensation_amount}', using 0")
            amount = 0.0

        # Outcome 2: Apology without compensation
        if not comp_yes:
            if amount != 0:
                print(f"Warning: compensation_decision=NO but amount={amount}. Forcing amount=0")
            return "NO", 0.0, compensation_reason

        # Outcome 3: Apology with compensation - validate amount
        if amount <= 0:
            print(f"Warning: compensation_decision=YES but amount={amount}. Must be positive. Using 0 and setting decision=NO")
            return "NO", 0.0, compensation_reason

        if amount > max_penalty:
            print(f"Warning: Compensation {amount} > max_penalty {max_penalty}. Capping at {max_penalty}")
            amount = max_penalty

        if amount > opponent_penalty:
            print(f"Warning: Compensation {amount} > opponent's penalty {opponent_penalty}. Capping at {opponent_penalty}")
            amount = opponent_penalty

        return "YES", amount, compensation_reason

    def _apply_compensation_transfers(self):
        """
        Apply compensation transfers after both agents decide.

        Transfer model:
        - Sender cumulative += compensation_amount
        - Recipient cumulative -= compensation_amount

        Validation:
        - Ensures no negative adjusted penalties result from transfers
        """
        compensation_cfg = self.game.config.get("apology", {}).get("compensation", {})
        if not compensation_cfg.get("enabled", False):
            for agent in self.game.agents.values():
                agent.add_adjusted_score(float(agent.last_score()))
            return

        agents_list = list(self.game.agents.values())
        if len(agents_list) != 2:
            raise ValueError("Compensation transfers require exactly 2 agents")

        agent1, agent2 = agents_list

        comp1 = agent1.last_compensation_amount()
        comp2 = agent2.last_compensation_amount()

        # Net transfer: paid - received (keep as float, no rounding)
        agent1_adjusted = float(agent1.last_score()) + (comp1 - comp2)
        agent2_adjusted = float(agent2.last_score()) + (comp2 - comp1)

        # Validate: ensure no negative adjusted penalties
        if agent1_adjusted < 0:
            print(f"ERROR: {agent1.name} adjusted penalty would be negative ({agent1_adjusted}). "
                  f"Score={agent1.last_score()}, pays={comp1}, receives={comp2}")
            raise ValueError(f"Invalid compensation: {agent1.name} would have negative adjusted penalty")

        if agent2_adjusted < 0:
            print(f"ERROR: {agent2.name} adjusted penalty would be negative ({agent2_adjusted}). "
                  f"Score={agent2.last_score()}, pays={comp2}, receives={comp1}")
            raise ValueError(f"Invalid compensation: {agent2.name} would have negative adjusted penalty")

        agent1.add_adjusted_score(agent1_adjusted)
        agent2.add_adjusted_score(agent2_adjusted)

        print(f"Compensation: {agent1.name} pays {comp1}, receives {comp2}")
        print(f"Compensation: {agent2.name} pays {comp2}, receives {comp1}")
        print(f"Adjusted penalties: {agent1.name}={agent1_adjusted}, {agent2.name}={agent2_adjusted}")

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

