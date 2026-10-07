class GoalProvider:

    def __init__(self, goals_config, goal_condition):
        if goal_condition not in goals_config:
            raise ValueError(
                f"Goal condition '{goal_condition}' not found in goals config. "
                f"Available: {list(goals_config.keys())}"
            )
        self._goal_text = goals_config[goal_condition]

    @property
    def goal_text(self):
        return self._goal_text
