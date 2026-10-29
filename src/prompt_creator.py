import re


class PromptCreator:

    def __init__(self, lang, prompt_template, n_rounds, n_rounds_known, payoff_matrix, config, goal_provider=None):
        self.language = lang
        self.prompt_template = prompt_template
        self.config = config
        self.n_rounds = n_rounds
        self.n_rounds_known = n_rounds_known
        self.payoff_matrix = payoff_matrix
        self.goal_provider = goal_provider

    def _find_part(self, field_name):
        pattern = rf"\{{{field_name}\}}:\s*\[(.*?)\]"
        return re.search(pattern, self.prompt_template, flags=re.DOTALL)

    def _remove_part(self, part):
        if part:
            self.prompt_template = self.prompt_template.replace(part.group(0), '')

    def _replace_part(self, part, replacement=None):
        if part:
            if replacement is not None:
                self.prompt_template = self.prompt_template.replace(
                    part.group(0),
                    replacement
                )
            else:
                self.prompt_template = self.prompt_template.replace(
                    part.group(0),
                    part.group(1)
                )

    def process_intro(self, agent, pv_dict):
        intro = self._find_part('intro')
        if intro is None:
            return

        if agent.personality == 'None':
            self._remove_part(intro)
        else:
            self._replace_part(intro)
            pv_dict['personality'] = agent.personality

    def process_opponent_intro(self, agent, opponents, pv_dict):
        opponent_intro = self._find_part('opponentIntro')
        if opponent_intro is None:
            return

        valid_opponents_exist = any(
            (opp.opponent_personality_prob != 0 and opp.personality != 'None')
            for opp in opponents
        )

        if not valid_opponents_exist:
            self._remove_part(opponent_intro)
        else:
            self._replace_part(opponent_intro)

            for i, opp in enumerate(opponents, start=1):
                pv_dict[f"opponent{i}"] = opp.name
                pv_dict[f"opponentPersonality{i}"] = opp.personality
                pv_dict[f"opponentPersonalityProbability{i}"] = opp.opponent_personality_prob

    def process_game_length(self, pv_dict):
        game_length = self._find_part('gameLength')
        if game_length is None:
            return

        if self.n_rounds_known:
            self._replace_part(game_length)
            pv_dict['nRounds'] = self.n_rounds
        else:
            self._remove_part(game_length)

    def map_placeholders(self, agent_name, opponents, current_round, history):
        strategies_keys = list(self.payoff_matrix.strategies.keys())
        weight_keys = list(self.payoff_matrix.weights.keys())

        values = {
            'currentPlayerName': agent_name,
            'currentRound': current_round,
            'history': history,
        }

        for i, key in enumerate(strategies_keys):
            values[f"strategy{i+1}"] = self.payoff_matrix.strategies[key]

        for i, key in enumerate(weight_keys):
            values[f"weight{i+1}"] = self.payoff_matrix.weights[key]

        for i, opp in enumerate(opponents, start=1):
            values[f"opponent{i}"] = opp.name
    
        #  ADD THIS BLOCK 
        lang = self.language
        # apology_map = self.config["apology"]["decisions"][lang]

        # values["apologyDecision1"] = apology_map["apologyDecision1"]
        # values["apologyDecision2"] = apology_map["apologyDecision2"]
        # values["apologyMessage"] = "apology_message"

        apology_map = self.config.get("apology", {}).get("decisions", {}).get(lang, {})

        values["apologyDecision1"] = apology_map.get("apologyDecision1", "YES")
        values["apologyDecision2"] = apology_map.get("apologyDecision2", "NO")

        return values

    def process_apology_history(self, pv_dict):
        apology_history = self._find_part('apologyHistory')
        if apology_history is None:
            return

        content = pv_dict.get("apologyHistoryContent", "")
        if content:
            self._replace_part(apology_history)
        else:
            self._remove_part(apology_history)

    def process_reason_codes(self, pv_dict):
        reason_codes_part = self._find_part('reasonCodes')
        if reason_codes_part is None:
            return

        include = (
            self.config
            .get("apology", {})
            .get("includeReasonCodeInNextAction", True)
        )

        if include:
            self._replace_part(reason_codes_part)
        else:
            self._remove_part(reason_codes_part)

    def process_optional_parts(self, agent, opponents, pv_dict):
        self.process_intro(agent, pv_dict)
        self.process_opponent_intro(agent, opponents, pv_dict)
        self.process_game_length(pv_dict)

    def _add_apology_labels(self, pv_dict):
        apology_cfg = self.config.get("apology", {})
        decisions = (
            apology_cfg.get("decisions", {}).get(self.language, {})
        )

        pv_dict["apologyYes"] = decisions.get("apologyDecision1", "YES")
        pv_dict["apologyNo"] = decisions.get("apologyDecision2", "NO")
        pv_dict["maxMessageLength"] = apology_cfg.get(
            "maxMessageLength", 25
        )

        # Add compensation placeholders
        comp_cfg = apology_cfg.get("compensation", {})
        pv_dict["maxCompensationReasonLength"] = comp_cfg.get("maxCompensationReasonLength", 30)
        pv_dict["maxPenalty"] = max(self.payoff_matrix.weights.values())

    def _add_outcome_placeholders(self, agent, opponents, pv_dict):
        pv_dict["playerAction"] = agent.last_strategy()
        pv_dict["playerPenalty"] = agent.last_score()
        pv_dict["playerAdjustedPenalty"] = agent.last_adjusted_score()

        opponent = opponents[0] if opponents else None
        if opponent is not None:
            pv_dict["opponentAction"] = opponent.last_strategy()
            pv_dict["opponentPenalty"] = opponent.last_score()
            pv_dict["opponentAdjustedPenalty"] = opponent.last_adjusted_score()
        else:
            pv_dict["opponentAction"] = ""
            pv_dict["opponentPenalty"] = ""
            pv_dict["opponentAdjustedPenalty"] = ""

    def _add_apology_placeholders(self, agent, opponents, pv_dict):
        self._add_apology_labels(pv_dict)
        self._add_outcome_placeholders(agent, opponents, pv_dict)

    def _format_apology_history(self, history, agent_name, opponents):
        if not history:
            return ""

        opponent_name = opponents[0].name if opponents else "opponent"
        paragraphs = []

        sorted_keys = sorted(
            history.keys(),
            key=lambda k: int(k.split("_")[1])
        )

        for round_key in sorted_keys:
            agents_data = history[round_key]
            round_num = int(round_key.split("_")[1])

            player_data = agents_data.get(agent_name, {})
            opponent_data = agents_data.get(opponent_name, {})

            your_action = player_data.get("strategy", "unknown")
            opp_action = opponent_data.get("strategy", "unknown")
            your_payoff = player_data.get("score", "?")
            opp_payoff = opponent_data.get("score", "?")

            n_rounds = len(sorted_keys)
            if n_rounds == 1:
                round_label = "In the previous round"
            else:
                round_label = f"In round {round_num}"

            lines = [
                f"{round_label}, you chose {your_action} "
                f"and your opponent chose {opp_action}. "
                f"Your payoff was {your_payoff} and your opponent's payoff was {opp_payoff}."
            ]

            p_apology = player_data.get("apology_decision")
            o_apology = opponent_data.get("apology_decision")
            p_comp = player_data.get("compensation_amount", 0)
            o_comp = opponent_data.get("compensation_amount", 0)

            if p_apology or o_apology:
                apology_parts = []
                if p_apology:
                    label = "apologised" if p_apology in ("apologize", "YES") else "did not apologise"
                    apology_line = f"Your apology decision: you {label}."
                    if p_comp > 0:
                        apology_line += f" You offered compensation of {p_comp}."
                    apology_parts.append(apology_line)
                if o_apology:
                    label = "apologised" if o_apology in ("apologize", "YES") else "did not apologise"
                    apology_line = f"Opponent's apology decision: opponent {label}."
                    if o_comp > 0:
                        apology_line += f" Opponent offered compensation of {o_comp}."
                    apology_parts.append(apology_line)
                lines.append(" ".join(apology_parts))

            # Add adjusted penalties if they exist
            player_adjusted = player_data.get("adjusted_score")
            opponent_adjusted = opponent_data.get("adjusted_score")
            if player_adjusted is not None or opponent_adjusted is not None:
                lines.append(
                    f"Adjusted penalties: yours={player_adjusted}, opponent's={opponent_adjusted}."
                )

            paragraphs.append("\n".join(lines))

        return "\n\n".join(paragraphs)

    def _lookup_personality_definition(self, personality):
        definitions = (
            self.config
            .get("personalityDefinitions", {})
            .get(self.language, {})
        )
        return definitions.get(personality, "")

    def _add_controlled_placeholders(self, agent, opponents, pv_dict):
        self._add_apology_labels(pv_dict)
        self._add_outcome_placeholders(agent, opponents, pv_dict)

        pv_dict["personalityDefinition"] = (
            self._lookup_personality_definition(agent.personality)
        )

        opponent = opponents[0] if opponents else None
        if opponent is not None:
            pv_dict["opponentPersonalityDefinition"] = (
                self._lookup_personality_definition(opponent.personality)
            )
        else:
            pv_dict["opponentPersonalityDefinition"] = ""

        if self.goal_provider:
            pv_dict["goalCondition"] = self.goal_provider.goal_text

        pv_dict["totalRoundsDescription"] = (
            f"There are {self.n_rounds} rounds in total."
            if self.n_rounds_known
            else "The total number of rounds is unknown."
        )

        history = pv_dict.get("history", {})
        pv_dict["apologyHistoryContent"] = (
            self._format_apology_history(
                history, agent.name, opponents
            )
        )

        self.process_apology_history(pv_dict)

    def fill_template(self, agent, opponents, current_round, history, phase):
        placeholder_value_dict = self.map_placeholders(
            agent.name,
            opponents,
            current_round,
            history
        )

        self.process_optional_parts(agent, opponents, placeholder_value_dict)

        communicate_match = self._find_part('communicate')
        choose_match = self._find_part('choose')

        phase_actions = {
            'communicate': {'replace': communicate_match, 'remove': choose_match},
            'choose': {'replace': choose_match, 'remove': communicate_match}
        }

        if phase in phase_actions:
            actions = phase_actions[phase]

            if actions['replace'] is not None:
                self._replace_part(actions['replace'])

            if actions['remove'] is not None:
                self._remove_part(actions['remove'])

        if phase == "apology":
            self._add_apology_placeholders(
                agent,
                opponents,
                placeholder_value_dict
            )

        elif phase == "controlled":
            self._add_controlled_placeholders(
                agent,
                opponents,
                placeholder_value_dict
            )

        elif phase == "controlled_next_action":
            self._add_controlled_placeholders(
                agent,
                opponents,
                placeholder_value_dict
            )

            # Current player apology and compensation
            current_apology = agent.last_apology_decision()
            placeholder_value_dict["currentApologyDecision"] = current_apology

            # Only show compensation details if apology was given
            current_apologized = current_apology and current_apology.upper() in ("YES", "APOLOGIZE")
            if current_apologized:
                placeholder_value_dict["currentApologyMessage"] = (
                    agent.last_apology_message()
                )
                placeholder_value_dict["currentCompensationDecision"] = (
                    agent.last_compensation_decision()
                )
                placeholder_value_dict["currentCompensationAmount"] = (
                    agent.last_compensation_amount()
                )
                placeholder_value_dict["currentCompensationReason"] = (
                    agent.last_compensation_reason()
                )
            else:
                placeholder_value_dict["currentApologyMessage"] = ""
                placeholder_value_dict["currentCompensationDecision"] = ""
                placeholder_value_dict["currentCompensationAmount"] = ""
                placeholder_value_dict["currentCompensationReason"] = ""

            # Opponent apology and compensation
            opponent = opponents[0] if opponents else None
            if opponent is not None:
                opponent_apology = opponent.last_apology_decision()
                placeholder_value_dict["opponentApologyDecision"] = opponent_apology

                # Only show compensation details if opponent apologized
                opponent_apologized = opponent_apology and opponent_apology.upper() in ("YES", "APOLOGIZE")
                if opponent_apologized:
                    placeholder_value_dict["opponentApologyMessage"] = (
                        opponent.last_apology_message()
                    )
                    placeholder_value_dict["opponentCompensationDecision"] = (
                        opponent.last_compensation_decision()
                    )
                    placeholder_value_dict["opponentCompensationAmount"] = (
                        opponent.last_compensation_amount()
                    )
                    placeholder_value_dict["opponentCompensationReason"] = (
                        opponent.last_compensation_reason()
                    )
                else:
                    placeholder_value_dict["opponentApologyMessage"] = ""
                    placeholder_value_dict["opponentCompensationDecision"] = ""
                    placeholder_value_dict["opponentCompensationAmount"] = ""
                    placeholder_value_dict["opponentCompensationReason"] = ""

                placeholder_value_dict["currentReasonCode"] = (
                    agent.last_reason_code()
                )
                placeholder_value_dict["opponentReasonCode"] = (
                    opponent.last_reason_code()
                )
            else:
                placeholder_value_dict["opponentApologyDecision"] = ""
                placeholder_value_dict["opponentApologyMessage"] = ""
                placeholder_value_dict["opponentCompensationDecision"] = ""
                placeholder_value_dict["opponentCompensationAmount"] = ""
                placeholder_value_dict["opponentCompensationReason"] = ""
                placeholder_value_dict["currentReasonCode"] = ""
                placeholder_value_dict["opponentReasonCode"] = ""

            # Process conditional compensation sections
            your_comp_details = self._find_part('yourCompensationDetails')
            if current_apologized and your_comp_details:
                self._replace_part(your_comp_details)
            elif your_comp_details:
                self._remove_part(your_comp_details)

            opp_comp_details = self._find_part('opponentCompensationDetails')
            opponent_apologized = (opponent and opponent.last_apology_decision()
                                  and opponent.last_apology_decision().upper() in ("YES", "APOLOGIZE"))
            if opponent_apologized and opp_comp_details:
                self._replace_part(opp_comp_details)
            elif opp_comp_details:
                self._remove_part(opp_comp_details)

            self.process_reason_codes(placeholder_value_dict)

        if self.goal_provider and "goalCondition" not in placeholder_value_dict:
            placeholder_value_dict["goalCondition"] = (
                self.goal_provider.goal_text
            )

        prompt = self.prompt_template.format(**placeholder_value_dict)

        phase_labels = {
            "controlled": "Apology Phase",
            "controlled_next_action": "Next Action Phase",
            "apology": "Apology Phase",
            "choose": "Strategy Phase",
            "communicate": "Communication Phase",
        }
        phase_label = phase_labels.get(phase, phase)
        print(f"\nCURRENT PROMPT - {phase_label}:\n{prompt}")

        return prompt