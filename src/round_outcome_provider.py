from abc import ABC, abstractmethod


class RoundOutcomeProvider(ABC):

    @abstractmethod
    def get_outcome(self, round_number, agents, payoff_matrix, game_round=None):
        """
        Return the outcome for a single round.

        Returns:
            dict: {agent_name: {"strategy": str, "score": int}, ...}
        """


class InteractiveOutcomeProvider(RoundOutcomeProvider):

    def get_outcome(self, round_number, agents, payoff_matrix, game_round=None):
        round_strategies = game_round.run_strategy_stage()

        payoff_matrix.attribute_scores(
            list(agents.values()),
            round_strategies
        )

        agent_names = list(agents.keys())
        strategy_values = [
            agents[name].last_strategy()
            for name in agent_names
        ]
        scores = [
            agents[name].last_score()
            for name in agent_names
        ]

        return {
            "round_strategies": round_strategies,
            "outcomes": {
                name: {
                    "strategy": strategy_values[i],
                    "score": scores[i]
                }
                for i, name in enumerate(agent_names)
            }
        }


class ControlledOutcomeProvider(RoundOutcomeProvider):

    def __init__(self, rounds_config, strategies_dict):
        self.rounds_config = rounds_config
        self.strategies_dict = strategies_dict

    def get_outcome(self, round_number, agents, payoff_matrix, game_round=None):
        round_index = round_number - 1
        round_cfg = self.rounds_config[round_index]

        agent_names = list(agents.keys())
        agent1_name = agent_names[0]
        agent2_name = agent_names[1]

        player_strategy_key = round_cfg["playerStrategy"]
        opponent_strategy_key = round_cfg["opponentStrategy"]

        player_strategy = self.strategies_dict[player_strategy_key]
        opponent_strategy = self.strategies_dict[opponent_strategy_key]

        player_penalty = round_cfg["playerPenalty"]
        opponent_penalty = round_cfg["opponentPenalty"]

        agents[agent1_name].add_strategy(player_strategy)
        agents[agent1_name].add_score(player_penalty)

        agents[agent2_name].add_strategy(opponent_strategy)
        agents[agent2_name].add_score(opponent_penalty)

        round_strategies = [
            player_strategy_key,
            opponent_strategy_key
        ]

        return {
            "round_strategies": round_strategies,
            "outcomes": {
                agent1_name: {
                    "strategy": player_strategy,
                    "score": player_penalty
                },
                agent2_name: {
                    "strategy": opponent_strategy,
                    "score": opponent_penalty
                }
            }
        }
