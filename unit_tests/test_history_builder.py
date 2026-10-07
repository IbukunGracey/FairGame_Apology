import unittest

from src.history_builder import HistoryBuilder


def _make_history(n_rounds):
    """Build a sample history dict with n_rounds completed rounds."""
    history = {}
    for r in range(1, n_rounds + 1):
        history[f"round_{r}"] = {
            "agent1": {
                "strategy": "Cooperate" if r % 2 else "Betray",
                "score": 1 if r % 2 else 3,
                "apology_decision": "YES",
                "apology_message": f"Sorry round {r}",
                "reason_code": "guilt",
            },
            "agent2": {
                "strategy": "Betray" if r % 2 else "Cooperate",
                "score": 5 if r % 2 else 1,
                "apology_decision": "NO",
                "apology_message": "",
                "reason_code": "",
            },
        }
    return history


class TestHistoryBuilderValidation(unittest.TestCase):

    def test_invalid_condition_raises(self):
        with self.assertRaises(ValueError):
            HistoryBuilder("bad_condition")

    def test_all_valid_conditions_accepted(self):
        for cond in ("one_shot", "no_memory", "memory1", "memory2", "memory_all"):
            hb = HistoryBuilder(cond)
            self.assertEqual(hb.condition, cond)


class TestOneShotCondition(unittest.TestCase):

    def setUp(self):
        self.builder = HistoryBuilder("one_shot")
        self.history = _make_history(4)

    def test_returns_empty_at_round_1(self):
        self.assertEqual(self.builder.build(self.history, 1), {})

    def test_returns_empty_at_round_4(self):
        self.assertEqual(self.builder.build(self.history, 4), {})


class TestNoMemoryCondition(unittest.TestCase):

    def setUp(self):
        self.builder = HistoryBuilder("no_memory")
        self.history = _make_history(4)

    def test_returns_empty_at_round_1(self):
        self.assertEqual(self.builder.build(self.history, 1), {})

    def test_returns_empty_at_round_2(self):
        self.assertEqual(self.builder.build(self.history, 2), {})

    def test_returns_empty_at_round_4(self):
        self.assertEqual(self.builder.build(self.history, 4), {})

    def test_no_strategy_leakage(self):
        result = self.builder.build(self.history, 4)
        self.assertNotIn("round_1", result)
        self.assertNotIn("round_2", result)
        self.assertNotIn("round_3", result)

    def test_no_apology_leakage(self):
        result = self.builder.build(self.history, 4)
        self.assertEqual(result, {})


class TestMemory1Condition(unittest.TestCase):

    def setUp(self):
        self.builder = HistoryBuilder("memory1")
        self.history = _make_history(4)

    def test_round_1_sees_nothing(self):
        result = self.builder.build(self.history, 1)
        self.assertEqual(result, {})

    def test_round_2_sees_round_1_only(self):
        result = self.builder.build(self.history, 2)
        self.assertIn("round_1", result)
        self.assertEqual(len(result), 1)

    def test_round_4_sees_round_3_only(self):
        result = self.builder.build(self.history, 4)
        self.assertIn("round_3", result)
        self.assertNotIn("round_1", result)
        self.assertNotIn("round_2", result)
        self.assertEqual(len(result), 1)

    def test_includes_apology_data(self):
        result = self.builder.build(self.history, 2)
        agent1_data = result["round_1"]["agent1"]
        self.assertIn("apology_decision", agent1_data)
        self.assertIn("apology_message", agent1_data)


class TestMemory2Condition(unittest.TestCase):

    def setUp(self):
        self.builder = HistoryBuilder("memory2")
        self.history = _make_history(4)

    def test_round_1_sees_nothing(self):
        result = self.builder.build(self.history, 1)
        self.assertEqual(result, {})

    def test_round_2_sees_round_1_only(self):
        result = self.builder.build(self.history, 2)
        self.assertIn("round_1", result)
        self.assertEqual(len(result), 1)

    def test_round_3_sees_rounds_1_and_2(self):
        result = self.builder.build(self.history, 3)
        self.assertIn("round_1", result)
        self.assertIn("round_2", result)
        self.assertEqual(len(result), 2)

    def test_round_4_sees_rounds_2_and_3_only(self):
        result = self.builder.build(self.history, 4)
        self.assertIn("round_2", result)
        self.assertIn("round_3", result)
        self.assertNotIn("round_1", result)
        self.assertEqual(len(result), 2)


class TestMemoryAllCondition(unittest.TestCase):

    def setUp(self):
        self.builder = HistoryBuilder("memory_all")
        self.history = _make_history(4)

    def test_returns_full_history(self):
        result = self.builder.build(self.history, 5)
        self.assertEqual(result, self.history)

    def test_includes_all_rounds(self):
        result = self.builder.build(self.history, 5)
        for r in range(1, 5):
            self.assertIn(f"round_{r}", result)

    def test_includes_apology_data(self):
        result = self.builder.build(self.history, 5)
        agent1_r1 = result["round_1"]["agent1"]
        self.assertEqual(agent1_r1["apology_decision"], "YES")
        self.assertEqual(agent1_r1["apology_message"], "Sorry round 1")


if __name__ == "__main__":
    unittest.main()
