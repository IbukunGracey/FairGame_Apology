import unittest
from src.scenario_generator import ScenarioGenerator


PAYOFF_MATRIX = {
    "weights": {
        "weight1": 6,
        "weight2": 10,
        "weight3": 0,
        "weight4": 2
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
}


class TestScenarioGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = ScenarioGenerator(PAYOFF_MATRIX, "en")

    def test_outcome_map_built_correctly(self):
        om = self.generator.outcome_map

        self.assertEqual(om["R"]["player_penalty"], 2)
        self.assertEqual(om["R"]["opponent_penalty"], 2)

        self.assertEqual(om["S"]["player_penalty"], 10)
        self.assertEqual(om["S"]["opponent_penalty"], 0)

        self.assertEqual(om["T"]["player_penalty"], 0)
        self.assertEqual(om["T"]["opponent_penalty"], 10)

        self.assertEqual(om["P"]["player_penalty"], 6)
        self.assertEqual(om["P"]["opponent_penalty"], 6)

    def test_cooperate_defect_identification(self):
        # strategy2 (OptionB) should be cooperate (lower symmetric penalty = 2)
        # strategy1 (OptionA) should be defect (higher symmetric penalty = 6)
        self.assertEqual(self.generator._cooperate_key, "strategy2")
        self.assertEqual(self.generator._defect_key, "strategy1")

    def test_no_memory_generates_4_scenarios(self):
        cfg = {"memoryType": "no_memory", "outcomes": ["R", "S", "T", "P"]}
        scenarios = self.generator.generate(cfg)

        self.assertEqual(len(scenarios), 4)

    def test_no_memory_scenario_ids(self):
        cfg = {"memoryType": "no_memory", "outcomes": ["R", "S", "T", "P"]}
        scenarios = self.generator.generate(cfg)
        ids = [s["scenario_id"] for s in scenarios]

        self.assertEqual(ids, ["R", "S", "T", "P"])

    def test_no_memory_all_single_round(self):
        cfg = {"memoryType": "no_memory", "outcomes": ["R", "S", "T", "P"]}
        scenarios = self.generator.generate(cfg)

        for s in scenarios:
            self.assertEqual(s["n_rounds"], 1)
            self.assertEqual(len(s["rounds"]), 1)

    def test_no_memory_correct_payoffs(self):
        cfg = {"memoryType": "no_memory", "outcomes": ["R", "S", "T", "P"]}
        scenarios = self.generator.generate(cfg)

        payoff_map = {s["scenario_id"]: s for s in scenarios}

        self.assertEqual(payoff_map["R"]["player_penalty"], 2)
        self.assertEqual(payoff_map["S"]["player_penalty"], 10)
        self.assertEqual(payoff_map["T"]["player_penalty"], 0)
        self.assertEqual(payoff_map["P"]["player_penalty"], 6)

    def test_memory1_generates_64_scenarios(self):
        cfg = {
            "memoryType": "memory1",
            "outcomes": ["R", "S", "T", "P"],
            "previousOutcomes": ["R", "S", "T", "P"],
            "previousApologyStates": ["AA", "AN", "NA", "NN"]
        }
        scenarios = self.generator.generate(cfg)

        self.assertEqual(len(scenarios), 64)

    def test_memory1_unique_scenario_ids(self):
        cfg = {
            "memoryType": "memory1",
            "outcomes": ["R", "S", "T", "P"],
            "previousOutcomes": ["R", "S", "T", "P"],
            "previousApologyStates": ["AA", "AN", "NA", "NN"]
        }
        scenarios = self.generator.generate(cfg)
        ids = [s["scenario_id"] for s in scenarios]

        self.assertEqual(len(ids), len(set(ids)))

    def test_memory1_all_two_rounds(self):
        cfg = {
            "memoryType": "memory1",
            "outcomes": ["R", "S", "T", "P"],
            "previousOutcomes": ["R", "S", "T", "P"],
            "previousApologyStates": ["AA", "AN", "NA", "NN"]
        }
        scenarios = self.generator.generate(cfg)

        for s in scenarios:
            self.assertEqual(s["n_rounds"], 2)
            self.assertEqual(len(s["rounds"]), 2)

    def test_memory1_has_previous_round_data(self):
        cfg = {
            "memoryType": "memory1",
            "outcomes": ["R"],
            "previousOutcomes": ["P"],
            "previousApologyStates": ["AA"]
        }
        scenarios = self.generator.generate(cfg)

        self.assertEqual(len(scenarios), 1)
        s = scenarios[0]
        self.assertEqual(s["scenario_id"], "R_PAA")
        self.assertEqual(s["current_outcome"], "R")
        self.assertEqual(s["previous_outcome"], "P")
        self.assertEqual(s["previous_apology_state"], "AA")
        self.assertIn("previous_round_data", s)

        prev = s["previous_round_data"]
        self.assertEqual(prev["player_apology"], "apologize")
        self.assertEqual(prev["opponent_apology"], "apologize")

    def test_memory1_apology_state_mapping(self):
        cfg = {
            "memoryType": "memory1",
            "outcomes": ["R"],
            "previousOutcomes": ["R"],
            "previousApologyStates": ["AN", "NA", "NN"]
        }
        scenarios = self.generator.generate(cfg)

        state_map = {s["previous_apology_state"]: s for s in scenarios}

        self.assertEqual(
            state_map["AN"]["previous_round_data"]["player_apology"],
            "apologize"
        )
        self.assertEqual(
            state_map["AN"]["previous_round_data"]["opponent_apology"],
            "do_not_apologize"
        )
        self.assertEqual(
            state_map["NA"]["previous_round_data"]["player_apology"],
            "do_not_apologize"
        )
        self.assertEqual(
            state_map["NA"]["previous_round_data"]["opponent_apology"],
            "apologize"
        )
        self.assertEqual(
            state_map["NN"]["previous_round_data"]["player_apology"],
            "do_not_apologize"
        )
        self.assertEqual(
            state_map["NN"]["previous_round_data"]["opponent_apology"],
            "do_not_apologize"
        )

    def test_invalid_outcome_raises(self):
        cfg = {"memoryType": "no_memory", "outcomes": ["R", "X"]}

        with self.assertRaises(ValueError) as ctx:
            self.generator.generate(cfg)

        self.assertIn("X", str(ctx.exception))

    def test_invalid_apology_state_raises(self):
        cfg = {
            "memoryType": "memory1",
            "outcomes": ["R"],
            "previousOutcomes": ["R"],
            "previousApologyStates": ["XX"]
        }

        with self.assertRaises(ValueError) as ctx:
            self.generator.generate(cfg)

        self.assertIn("XX", str(ctx.exception))

    def test_unknown_memory_type_raises(self):
        cfg = {"memoryType": "memory3"}

        with self.assertRaises(NotImplementedError):
            self.generator.generate(cfg)

    def test_subset_of_outcomes(self):
        cfg = {"memoryType": "no_memory", "outcomes": ["R", "S"]}
        scenarios = self.generator.generate(cfg)

        self.assertEqual(len(scenarios), 2)
        ids = [s["scenario_id"] for s in scenarios]
        self.assertEqual(ids, ["R", "S"])

    def test_one_shot_same_as_no_memory(self):
        cfg = {"memoryType": "one_shot", "outcomes": ["R", "S", "T", "P"]}
        scenarios = self.generator.generate(cfg)

        self.assertEqual(len(scenarios), 4)
        for s in scenarios:
            self.assertEqual(s["memory_type"], "one_shot")

    def test_strategies_derived_from_matrix_not_hardcoded(self):
        cfg = {"memoryType": "no_memory", "outcomes": ["R"]}
        scenarios = self.generator.generate(cfg)

        s = scenarios[0]
        # R = both cooperate, which is strategy2 in this matrix
        self.assertEqual(s["player_strategy_key"], "strategy2")
        self.assertEqual(s["opponent_strategy_key"], "strategy2")


if __name__ == "__main__":
    unittest.main()
