import unittest
from unittest.mock import patch, MagicMock
import json

from src.fairgame_factory import FairGameFactory
from src.scenario_generator import ScenarioGenerator
from src.agent import Agent, _extract_json_object
from src.io_managers.configuration_validator import ConfigValidator


def _base_config(memory_type="no_memory", scenario_gen=True, personalities=None):
    """Build a minimal valid config for controlled scenario experiments."""
    if personalities is None:
        personalities = ["None", "None"]

    cfg = {
        "name": "Prisoner's Dilemma",
        "gamePromptFile": "prisoner_dilemma_en.txt",
        "nRounds": 1,
        "nRoundsIsKnown": True,
        "llm": "OpenAIGPT4o",
        "languages": ["en"],
        "allAgentPermutations": False,
        "agents": {
            "names": ["agent1", "agent2"],
            "personalities": {"en": personalities},
            "opponentPersonalityProb": [0, 0]
        },
        "payoffMatrix": {
            "weights": {
                "weight1": 6, "weight2": 10,
                "weight3": 0, "weight4": 2
            },
            "strategies": {
                "en": {
                    "strategy1": "OptionA",
                    "strategy2": "OptionB"
                }
            },
            "combinations": {
                "combination1": ["strategy1", "strategy1"],
                "combination2": ["strategy1", "strategy2"],
                "combination3": ["strategy2", "strategy1"],
                "combination4": ["strategy2", "strategy2"]
            },
            "matrix": {
                "combination1": ["weight1", "weight1"],
                "combination2": ["weight3", "weight2"],
                "combination3": ["weight2", "weight3"],
                "combination4": ["weight4", "weight4"]
            }
        },
        "personalityDefinitions": {
            "en": {
                "cooperative": "You value mutual benefit.",
                "selfish": "You prioritize yourself.",
                "None": ""
            }
        },
        "experiment": {
            "mode": "controlled",
            "scenarioGeneration": scenario_gen,
            "memoryType": memory_type,
            "outcomes": ["R", "S", "T", "P"],
            "previousOutcomes": ["R", "S", "T", "P"],
            "previousApologyStates": ["AA", "AN", "NA", "NN"],
            "goalCondition": "minimize_penalty"
        },
        "goals": {
            "minimize_penalty": "Minimize your penalty."
        },
        "apology": {
            "enabled": True,
            "promptFile": "prisoner_dilemma_apology_en.txt",
            "maxMessageLength": 30,
            "decisions": {
                "en": {
                    "apologyDecision1": "YES",
                    "apologyDecision2": "NO"
                }
            },
            "outputSchema": {
                "apology_decision": "string",
                "apology_message": "string",
                "reason_code": "string",
                "next_action": "string"
            },
            "includeReasonCodeInNextAction": True
        },
        "stopGameWhen": [],
        "agentsCommunicate": False,
        "promptTemplate": {"en": "dummy template {currentPlayerName} {strategy1} {strategy2} {currentRound} {history} {opponent1} {apologyDecision1} {apologyDecision2}"},
        "apologyPromptTemplate": {"en": "apology template {currentPlayerName} {playerAction} {opponentAction} {playerPenalty} {opponentPenalty} {apologyYes} {apologyNo} {maxMessageLength}"},
        "controlledPromptTemplate": {"en": "controlled template {currentPlayerName} {opponent1} {strategy1} {strategy2} {currentRound} {playerAction} {opponentAction} {playerPenalty} {opponentPenalty} {apologyYes} {apologyNo} {maxMessageLength} {goalCondition} {totalRoundsDescription} {history}"},
        "controlledNextActionPromptTemplate": {"en": "next action template {currentPlayerName} {opponent1} {strategy1} {strategy2} {currentRound} {playerAction} {opponentAction} {playerPenalty} {opponentPenalty} {goalCondition} {totalRoundsDescription} {currentApologyDecision} {currentApologyMessage} {opponentApologyDecision} {opponentApologyMessage} {currentReasonCode} {opponentReasonCode} {history}"}
    }
    return cfg


class TestControlledExperimentGameCreation(unittest.TestCase):

    def test_no_memory_creates_4_games(self):
        config = _base_config("no_memory")
        factory = FairGameFactory()
        games = factory.create_games(config)

        self.assertEqual(len(games), 4)

    def test_memory1_creates_64_games(self):
        config = _base_config("memory1")
        factory = FairGameFactory()
        games = factory.create_games(config)

        self.assertEqual(len(games), 64)

    def test_each_game_has_scenario_metadata(self):
        config = _base_config("no_memory")
        factory = FairGameFactory()
        games = factory.create_games(config)

        outcomes = set()
        for game in games:
            desc = game.description
            self.assertIn("scenario_id", desc)
            self.assertIn("current_outcome", desc)
            self.assertIn("memory_type", desc)
            outcomes.add(desc["current_outcome"])

        self.assertEqual(outcomes, {"R", "S", "T", "P"})

    def test_memory1_game_has_previous_fields(self):
        config = _base_config("memory1")
        config["experiment"]["outcomes"] = ["R"]
        config["experiment"]["previousOutcomes"] = ["P"]
        config["experiment"]["previousApologyStates"] = ["AA"]

        factory = FairGameFactory()
        games = factory.create_games(config)

        self.assertEqual(len(games), 1)
        desc = games[0].description
        self.assertEqual(desc["previous_outcome"], "P")
        self.assertEqual(desc["previous_apology_state"], "AA")

    def test_memory1_game_has_prepopulated_history(self):
        config = _base_config("memory1")
        config["experiment"]["outcomes"] = ["R"]
        config["experiment"]["previousOutcomes"] = ["P"]
        config["experiment"]["previousApologyStates"] = ["AN"]

        factory = FairGameFactory()
        games = factory.create_games(config)

        game = games[0]
        round_1 = game.history.get_round_data(1)
        self.assertIn("agent1", round_1)
        self.assertIn("agent2", round_1)

        agent1_data = round_1["agent1"]
        self.assertEqual(agent1_data["apology_decision"], "apologize")
        agent2_data = round_1["agent2"]
        self.assertEqual(agent2_data["apology_decision"], "do_not_apologize")

    def test_memory1_starts_at_round_2(self):
        """Memory1 games must skip round 1 execution — previous round is
        pre-populated, so the game loop should start at round 2."""
        config = _base_config("memory1")
        config["experiment"]["outcomes"] = ["R"]
        config["experiment"]["previousOutcomes"] = ["P"]
        config["experiment"]["previousApologyStates"] = ["AA"]

        factory = FairGameFactory()
        games = factory.create_games(config)

        game = games[0]
        self.assertEqual(game.current_round, 2,
            "memory1 game should start at round 2 (round 1 is predefined)")
        self.assertEqual(game.n_rounds, 2)
        self.assertIn("round_1", game.history.rounds,
            "round 1 history should be pre-populated")

    def test_no_memory_starts_at_round_1(self):
        """No-memory games should start at round 1 as normal."""
        config = _base_config("no_memory")
        factory = FairGameFactory()
        games = factory.create_games(config)

        for game in games:
            self.assertEqual(game.current_round, 1)

    def test_no_memory_games_are_independent(self):
        config = _base_config("no_memory")
        factory = FairGameFactory()
        games = factory.create_games(config)

        for game in games:
            self.assertEqual(game.n_rounds, 1)
            if game.scenario_metadata.get("memory_type") == "no_memory":
                self.assertEqual(len(game.history.rounds), 0)

    def test_personality_none_omitted_from_prompt(self):
        config = _base_config("no_memory")
        factory = FairGameFactory()
        games = factory.create_games(config)

        game = games[0]
        for agent in game.agents.values():
            self.assertEqual(agent.personality, "None")

    def test_personality_cooperative_included(self):
        config = _base_config(
            "no_memory",
            personalities=["cooperative", "selfish"]
        )
        factory = FairGameFactory()
        games = factory.create_games(config)

        game = games[0]
        agent_names = list(game.agents.keys())
        self.assertEqual(
            game.agents[agent_names[0]].personality,
            "cooperative"
        )
        self.assertEqual(
            game.agents[agent_names[1]].personality,
            "selfish"
        )

    def test_backward_compatible_without_scenario_gen(self):
        config = _base_config(scenario_gen=False)
        config["nRounds"] = 4
        config["experiment"]["rounds"] = [
            {
                "outcome": "R",
                "playerStrategy": "strategy2",
                "opponentStrategy": "strategy2",
                "playerPenalty": 2,
                "opponentPenalty": 2
            },
            {
                "outcome": "S",
                "playerStrategy": "strategy2",
                "opponentStrategy": "strategy1",
                "playerPenalty": 10,
                "opponentPenalty": 0
            },
            {
                "outcome": "T",
                "playerStrategy": "strategy1",
                "opponentStrategy": "strategy2",
                "playerPenalty": 0,
                "opponentPenalty": 10
            },
            {
                "outcome": "P",
                "playerStrategy": "strategy1",
                "opponentStrategy": "strategy1",
                "playerPenalty": 6,
                "opponentPenalty": 6
            }
        ]
        factory = FairGameFactory()
        games = factory.create_games(config)

        # Without scenarioGeneration, should create 1 game with 4 rounds
        self.assertEqual(len(games), 1)
        self.assertEqual(games[0].n_rounds, 4)


class TestApologyNextActionParsing(unittest.TestCase):

    def test_extract_next_action_from_json(self):
        response_text = json.dumps({
            "apology_decision": "YES",
            "apology_message": "I am sorry.",
            "reason_code": "remorse",
            "next_action": "OptionB"
        })

        parsed = _extract_json_object(response_text)

        self.assertEqual(parsed["apology_decision"], "YES")
        self.assertEqual(parsed["next_action"], "OptionB")

    def test_agent_execute_apology_returns_next_action(self):
        agent = Agent(
            name="test_agent",
            llm_service="OpenAIGPT4o",
            personality="None",
            opponent_personality_prob=0
        )

        mock_response = json.dumps({
            "apology_decision": "YES",
            "apology_message": "Sorry for that.",
            "reason_code": "repair_trust",
            "next_action": "OptionA"
        })

        with patch(
            "src.agent.execute_prompt",
            return_value=mock_response
        ):
            result = agent.execute_apology_round("test prompt")

        self.assertEqual(result["apology_decision"], "YES")
        self.assertEqual(result["apology_message"], "Sorry for that.")
        self.assertEqual(result["reason_code"], "repair_trust")
        self.assertEqual(result["next_action"], "OptionA")

    def test_agent_next_action_list(self):
        agent = Agent(
            name="test_agent",
            llm_service="OpenAIGPT4o",
            personality="None",
            opponent_personality_prob=0
        )

        agent.add_next_action("OptionA")
        agent.add_next_action("OptionB")

        self.assertEqual(agent.last_next_action(), "OptionB")
        self.assertEqual(len(agent.next_actions), 2)

    def test_agent_last_next_action_empty(self):
        agent = Agent(
            name="test_agent",
            llm_service="OpenAIGPT4o",
            personality="None",
            opponent_personality_prob=0
        )

        self.assertEqual(agent.last_next_action(), "")


class TestConfigValidatorScenarios(unittest.TestCase):

    def test_valid_no_memory_config(self):
        validator = ConfigValidator()
        config = _base_config("no_memory")
        result = validator.validate_config_structure(config)
        self.assertIsNotNone(result)

    def test_valid_memory1_config(self):
        validator = ConfigValidator()
        config = _base_config("memory1")
        result = validator.validate_config_structure(config)
        self.assertIsNotNone(result)

    def test_invalid_memory_type_raises(self):
        validator = ConfigValidator()
        config = _base_config("no_memory")
        config["experiment"]["memoryType"] = "memory5"

        with self.assertRaises(ValueError):
            validator.validate_config_structure(config)

    def test_invalid_outcome_code_raises(self):
        validator = ConfigValidator()
        config = _base_config("no_memory")
        config["experiment"]["outcomes"] = ["R", "X"]

        with self.assertRaises(ValueError):
            validator.validate_config_structure(config)

    def test_invalid_apology_state_raises(self):
        validator = ConfigValidator()
        config = _base_config("memory1")
        config["experiment"]["previousApologyStates"] = ["AB"]

        with self.assertRaises(ValueError):
            validator.validate_config_structure(config)


class TestResultsStructure(unittest.TestCase):

    def test_game_data_includes_scenario_metadata(self):
        from src.results_processing.game_data import GameData
        from src.results_processing.agent_info import AgentInfo

        agents = [
            AgentInfo("agent1", "OpenAIGPT4o", "None", 0),
            AgentInfo("agent2", "OpenAIGPT4o", "None", 0)
        ]
        round_data = {
            "agent1": {
                "strategies": ["OptionB"],
                "scores": [2],
                "messages": [],
                "apology_decisions": ["YES"],
                "apology_messages": ["Sorry"],
                "reason_codes": ["remorse"],
                "next_actions": ["OptionA"]
            },
            "agent2": {
                "strategies": ["OptionB"],
                "scores": [2],
                "messages": [],
                "apology_decisions": ["NO"],
                "apology_messages": [""],
                "reason_codes": ["no_responsibility"],
                "next_actions": ["OptionB"]
            }
        }

        gd = GameData(
            game_id="game_0",
            language="en",
            n_rounds=1,
            n_rounds_is_known=True,
            agents_communicate=False,
            agents=agents,
            agents_round_data=round_data,
            experiment_mode="controlled",
            scenario_metadata={
                "scenario_id": "R",
                "current_outcome": "R",
                "memory_type": "no_memory"
            }
        )

        d = gd.to_dict()

        self.assertEqual(d["scenario_id"], "R")
        self.assertEqual(d["current_outcome"], "R")
        self.assertEqual(d["memory_type"], "no_memory")
        self.assertNotIn("previous_outcome", d)
        self.assertEqual(d["agent1_next_actions"], ["OptionA"])
        self.assertEqual(d["agent2_next_actions"], ["OptionB"])

    def test_game_data_memory1_includes_previous_fields(self):
        from src.results_processing.game_data import GameData
        from src.results_processing.agent_info import AgentInfo

        agents = [
            AgentInfo("agent1", "OpenAIGPT4o", "None", 0),
            AgentInfo("agent2", "OpenAIGPT4o", "None", 0)
        ]
        round_data = {
            "agent1": {
                "strategies": ["OptionA", "OptionB"],
                "scores": [6, 2],
                "messages": [],
                "apology_decisions": ["NO", "YES"],
                "apology_messages": ["", "Sorry"],
                "reason_codes": ["", "remorse"],
                "next_actions": [None, "OptionB"]
            },
            "agent2": {
                "strategies": ["OptionA", "OptionB"],
                "scores": [6, 2],
                "messages": [],
                "apology_decisions": ["YES", "NO"],
                "apology_messages": ["My apologies", ""],
                "reason_codes": ["politeness", ""],
                "next_actions": [None, "OptionA"]
            }
        }

        gd = GameData(
            game_id="game_0",
            language="en",
            n_rounds=2,
            n_rounds_is_known=True,
            agents_communicate=False,
            agents=agents,
            agents_round_data=round_data,
            experiment_mode="controlled",
            scenario_metadata={
                "scenario_id": "R_PAA",
                "current_outcome": "R",
                "memory_type": "memory1",
                "previous_outcome": "P",
                "previous_apology_state": "AA"
            }
        )

        d = gd.to_dict()

        self.assertEqual(d["scenario_id"], "R_PAA")
        self.assertEqual(d["previous_outcome"], "P")
        self.assertEqual(d["previous_apology_state"], "AA")


class TestAgentLastReasonCode(unittest.TestCase):

    def test_last_reason_code_returns_most_recent(self):
        agent = Agent(
            name="test_agent",
            llm_service="OpenAIGPT4o",
            personality="None",
            opponent_personality_prob=0
        )
        agent.add_apology("YES", "Sorry", "remorse")
        agent.add_apology("NO", "", "no_responsibility")
        self.assertEqual(agent.last_reason_code(), "no_responsibility")

    def test_last_reason_code_empty_when_no_apologies(self):
        agent = Agent(
            name="test_agent",
            llm_service="OpenAIGPT4o",
            personality="None",
            opponent_personality_prob=0
        )
        self.assertEqual(agent.last_reason_code(), "")


class TestNextActionPromptContent(unittest.TestCase):

    def _make_prompt(self, include_reason_code=True, use_conditional_template=False):
        from src.prompt_creator import PromptCreator
        from src.payoff_matrix import PayoffMatrix
        from src.goal_provider import GoalProvider

        config = _base_config("no_memory")
        config["apology"]["includeReasonCodeInNextAction"] = include_reason_code

        if use_conditional_template:
            template = (
                "{currentPlayerName} "
                "{opponent1} {strategy1} {strategy2} {currentRound} "
                "{playerAction} {opponentAction} {playerPenalty} {opponentPenalty} "
                "{goalCondition} {totalRoundsDescription} "
                "own_decision={currentApologyDecision} "
                "own_message={currentApologyMessage} "
                "opp_decision={opponentApologyDecision} "
                "opp_message={opponentApologyMessage} "
                "{reasonCodes}: [own_reason={currentReasonCode} opp_reason={opponentReasonCode}] "
                "{history}"
            )
        else:
            template = config["controlledNextActionPromptTemplate"]["en"]

        agent = Agent("agent1", "OpenAIGPT4o", "None", 0)
        opponent = Agent("agent2", "OpenAIGPT4o", "None", 0)

        agent.add_strategy("OptionB")
        agent.add_score(2)
        agent.add_apology("YES", "I am sorry.", "remorse")

        opponent.add_strategy("OptionB")
        opponent.add_score(2)
        opponent.add_apology("NO", "", "no_responsibility")

        pm = PayoffMatrix(config["payoffMatrix"], "en")
        goal_provider = GoalProvider(config["goals"], "minimize_penalty")

        creator = PromptCreator(
            "en", template, 1, True, pm, config,
            goal_provider=goal_provider
        )

        prompt = creator.fill_template(
            agent, [opponent], 1, {}, "controlled_next_action"
        )
        return prompt

    def test_next_action_includes_opponent_apology(self):
        prompt = self._make_prompt()
        self.assertIn("NO", prompt)
        self.assertIn("I am sorry.", prompt)
        self.assertIn("YES", prompt)

    def test_next_action_includes_own_apology_message(self):
        prompt = self._make_prompt()
        self.assertIn("I am sorry.", prompt)

    def test_reason_codes_included_when_flag_true(self):
        prompt = self._make_prompt(
            include_reason_code=True,
            use_conditional_template=True
        )
        self.assertIn("own_reason=remorse", prompt)
        self.assertIn("opp_reason=no_responsibility", prompt)

    def test_reason_codes_excluded_when_flag_false(self):
        prompt = self._make_prompt(
            include_reason_code=False,
            use_conditional_template=True
        )
        self.assertNotIn("own_reason=", prompt)
        self.assertNotIn("opp_reason=", prompt)
        self.assertNotIn("reasonCodes", prompt)

    def test_reason_codes_default_included_when_flag_absent(self):
        from src.prompt_creator import PromptCreator
        from src.payoff_matrix import PayoffMatrix
        from src.goal_provider import GoalProvider

        config = _base_config("no_memory")
        del config["apology"]["includeReasonCodeInNextAction"]

        template = (
            "{currentPlayerName} {opponent1} {strategy1} {strategy2} "
            "{currentRound} {playerAction} {opponentAction} "
            "{playerPenalty} {opponentPenalty} "
            "{goalCondition} {totalRoundsDescription} "
            "{currentApologyDecision} {currentApologyMessage} "
            "{opponentApologyDecision} {opponentApologyMessage} "
            "{reasonCodes}: [reason={currentReasonCode} opp_reason={opponentReasonCode}] "
            "{history}"
        )

        agent = Agent("agent1", "OpenAIGPT4o", "None", 0)
        opponent = Agent("agent2", "OpenAIGPT4o", "None", 0)
        agent.add_strategy("OptionB")
        agent.add_score(2)
        agent.add_apology("YES", "Sorry", "remorse")
        opponent.add_strategy("OptionB")
        opponent.add_score(2)
        opponent.add_apology("NO", "", "no_responsibility")

        pm = PayoffMatrix(config["payoffMatrix"], "en")
        goal_provider = GoalProvider(config["goals"], "minimize_penalty")
        creator = PromptCreator(
            "en", template, 1, True, pm, config,
            goal_provider=goal_provider
        )

        prompt = creator.fill_template(
            agent, [opponent], 1, {}, "controlled_next_action"
        )
        self.assertIn("reason=remorse", prompt)
        self.assertIn("opp_reason=no_responsibility", prompt)


if __name__ == "__main__":
    unittest.main()
