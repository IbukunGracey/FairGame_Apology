import itertools
import pandas as pd

from src.fairgame import FairGame
from src.io_managers.io_manager import IoManager
from src.round_outcome_provider import (
    InteractiveOutcomeProvider,
    ControlledOutcomeProvider
)
from src.history_builder import HistoryBuilder
from src.goal_provider import GoalProvider
from src.payoff_matrix import PayoffMatrix
from src.scenario_generator import ScenarioGenerator


class FairGameFactory:
    """
    Factory class responsible for:
    - generating agent permutations
    - building game configurations
    - creating FairGame instances
    - running and collecting outputs
    """

    def __init__(self):
        self.io_manager = IoManager()
        self.config_all_langs_df = pd.DataFrame()
        self.games = []
        self.output_dict = {}

    # =========================================================
    # CONFIG GENERATION
    # =========================================================

    def _generate_language_config_df(self, config, lang):
        if config["allAgentPermutations"]:
            return self.compute_all_game_configurations(
                lang,
                config["agents"],
                config["llm"]
            )

        return self.compute_configuration(
            lang,
            config["agents"],
            config["llm"]
        )

    def _compute_agent_configurations(self, lang, config_agents):
        n_agents = len(config_agents["names"])

        agent_combinations = [config_agents["names"]]

        personality_permutations = list(
            itertools.product(
                config_agents["personalities"][lang],
                repeat=n_agents
            )
        )

        knowledge_permutations = list(
            itertools.product(
                config_agents["opponentPersonalityProb"],
                repeat=n_agents
            )
        )

        return (
            agent_combinations,
            personality_permutations,
            knowledge_permutations
        )

    def _generate_full_permutations(
        self,
        agent_combinations,
        personality_permutations,
        knowledge_permutations
    ):
        rows = []

        for agents in agent_combinations:
            n_agents = len(agents)

            for personality_tuple, knowledge_tuple in itertools.product(
                personality_permutations,
                knowledge_permutations
            ):
                row_dict = {
                    **{
                        f"Agent{i+1}": agents[i]
                        for i in range(n_agents)
                    },
                    **{
                        f"Personality{i+1}": personality_tuple[i]
                        for i in range(n_agents)
                    },
                    **{
                        f"OpponentPersonalityProb{i+1}": knowledge_tuple[i]
                        for i in range(n_agents)
                    }
                }
                rows.append(row_dict)

        return pd.DataFrame(rows)

    def compute_all_game_configurations(self, lang, config_agents, llm_service):
        agent_combinations, pers, knowledge = self._compute_agent_configurations(
            lang,
            config_agents
        )

        df = self._generate_full_permutations(
            agent_combinations,
            pers,
            knowledge
        )

        df["LLM"] = llm_service
        df["Language"] = lang

        return df

    def compute_configuration(self, lang, config_agents, llm_service):
        n_agents = len(config_agents["names"])

        row_dict = {}

        for i in range(n_agents):
            row_dict[f"Agent{i+1}"] = config_agents["names"][i]
            row_dict[f"Personality{i+1}"] = config_agents["personalities"][lang][i]
            row_dict[f"OpponentPersonalityProb{i+1}"] = config_agents["opponentPersonalityProb"][i]

        row_dict["LLM"] = llm_service
        row_dict["Language"] = lang

        return pd.DataFrame([row_dict])

    # =========================================================
    # GAME CREATION
    # =========================================================

    def _create_single_game(self, config, game_config_row, payoff_matrix):
        lang = game_config_row["Language"]

        experiment_cfg = config.get("experiment", {})
        mode = experiment_cfg.get("mode", "interactive")

        game_prompt_template = self.build_prompt_template(
            config, lang, stage="game"
        )

        apology_prompt_template = self.build_prompt_template(
            config, lang, stage="apology"
        )

        controlled_prompt_template = None
        controlled_next_action_template = None
        if mode == "controlled":
            controlled_prompt_template = self.build_prompt_template(
                config, lang, stage="controlled"
            )
            controlled_next_action_template = self.build_prompt_template(
                config, lang, stage="controlled_next_action"
            )

        agents = self.create_agents(game_config_row)

        pm_obj = PayoffMatrix(payoff_matrix, lang)

        if mode == "controlled":
            outcome_provider = ControlledOutcomeProvider(
                experiment_cfg["rounds"],
                pm_obj.strategies
            )
        else:
            outcome_provider = InteractiveOutcomeProvider()

        history_condition = experiment_cfg.get(
            "historyCondition", "memory_all"
        )
        history_builder = HistoryBuilder(history_condition)

        goal_provider = None
        goals_config = config.get("goals")
        goal_condition = experiment_cfg.get("goalCondition")
        if goals_config and goal_condition:
            goal_provider = GoalProvider(
                goals_config, goal_condition
            )

        return FairGame(
            config["name"],
            lang,
            agents,
            config["nRounds"],
            config["nRoundsIsKnown"],
            payoff_matrix,
            game_prompt_template,
            apology_prompt_template,
            config["stopGameWhen"],
            config["agentsCommunicate"],
            config,
            experiment_mode=mode,
            round_outcome_provider=outcome_provider,
            history_builder=history_builder,
            goal_provider=goal_provider,
            controlled_prompt_template=controlled_prompt_template,
            controlled_next_action_template=controlled_next_action_template
        )

    def create_agents(self, game_config_row):
        agents_dict = {}
        i = 1

        while f"Agent{i}" in game_config_row:
            agents_dict[game_config_row[f"Agent{i}"]] = {
                "name": game_config_row[f"Agent{i}"],
                "llm_service": game_config_row["LLM"],
                "personality": game_config_row[f"Personality{i}"],
                "opponent_personality_prob": game_config_row[f"OpponentPersonalityProb{i}"]
            }

            i += 1

        # convert dict → Agent objects (IMPORTANT FIX if your FairGame expects Agent class)
        from src.agent import Agent

        return {
            name: Agent(**data)
            for name, data in agents_dict.items()
        }

    # =========================================================
    # TEMPLATE HANDLING (UPDATED FIX HERE)
    # =========================================================

    def build_prompt_template(self, config, lang, stage="game"):

        if stage == "game":
            template = config.get("promptTemplate", {}).get(lang)

        elif stage == "apology":
            template = config.get("apologyPromptTemplate", {}).get(lang)

        elif stage == "controlled":
            template = config.get("controlledPromptTemplate", {}).get(lang)

        elif stage == "controlled_next_action":
            template = config.get("controlledNextActionPromptTemplate", {}).get(lang)

        else:
            raise ValueError(f"Unknown stage: {stage}")

        if template is None:
            template_filename = config.get("templateFilename")

            if stage == "apology":
                template_filename = config.get(
                    "apologyTemplateFilename",
                    template_filename
                )

            template = self.io_manager.load_stage_prompt(
                template_filename,
                lang,
                stage=stage
            )

        return template

    # =========================================================
    # OUTPUT HANDLING
    # =========================================================

    def _upload_output(self, game, game_history, game_n):
        game.description.pop("payoff_matrix", None)

        self.output_dict[f"game_{game_n}"] = {
            "description": game.description,
            "history": game_history.describe()
        }

    def set_io_manager(self, manager):
        self.io_manager = manager

    def results_games(self):
        return self.output_dict

    def all_game_configurations(self):
        return self.config_all_langs_df

    def load_config(self, config_filename):
        return self.io_manager.load_config(config_filename)

    # =========================================================
    # CONTROLLED EXPERIMENT (SCENARIO-BASED)
    # =========================================================

    def _create_controlled_experiment_games(self, config, game_config_row):
        """Create one FairGame per generated scenario for a single
        agent-personality permutation row."""

        lang = game_config_row["Language"]
        experiment_cfg = config.get("experiment", {})
        payoff_matrix_data = config["payoffMatrix"]

        generator = ScenarioGenerator(payoff_matrix_data, lang)
        scenarios = generator.generate(experiment_cfg)

        game_prompt_template = self.build_prompt_template(
            config, lang, stage="game"
        )
        apology_prompt_template = self.build_prompt_template(
            config, lang, stage="apology"
        )
        controlled_prompt_template = self.build_prompt_template(
            config, lang, stage="controlled"
        )
        controlled_next_action_template = self.build_prompt_template(
            config, lang, stage="controlled_next_action"
        )

        goal_provider = None
        goals_config = config.get("goals")
        goal_condition = experiment_cfg.get("goalCondition")
        if goals_config and goal_condition:
            goal_provider = GoalProvider(goals_config, goal_condition)

        games = []
        for scenario in scenarios:
            agents = self.create_agents(game_config_row)
            pm_obj = PayoffMatrix(payoff_matrix_data, lang)

            outcome_provider = ControlledOutcomeProvider(
                scenario["rounds"],
                pm_obj.strategies
            )

            history_builder = HistoryBuilder(
                scenario.get("history_condition", "no_memory")
            )

            scenario_metadata = {
                "scenario_id": scenario["scenario_id"],
                "current_outcome": scenario["current_outcome"],
                "memory_type": scenario["memory_type"],
            }
            if "previous_outcome" in scenario:
                scenario_metadata["previous_outcome"] = scenario["previous_outcome"]
            if "previous_apology_state" in scenario:
                scenario_metadata["previous_apology_state"] = scenario["previous_apology_state"]

            game = FairGame(
                config["name"],
                lang,
                agents,
                scenario["n_rounds"],
                config["nRoundsIsKnown"],
                payoff_matrix_data,
                game_prompt_template,
                apology_prompt_template,
                config["stopGameWhen"],
                config["agentsCommunicate"],
                config,
                experiment_mode="controlled",
                round_outcome_provider=outcome_provider,
                history_builder=history_builder,
                goal_provider=goal_provider,
                controlled_prompt_template=controlled_prompt_template,
                controlled_next_action_template=controlled_next_action_template,
                scenario_metadata=scenario_metadata
            )

            # For memory1: pre-populate synthetic previous-round history
            if scenario["memory_type"] == "memory1" and "previous_round_data" in scenario:
                prev = scenario["previous_round_data"]
                agent_names = list(agents.keys())
                agent1_name = agent_names[0]
                agent2_name = agent_names[1]

                game.history.update_round(1, agent1_name, {
                    "strategy": prev["player_action"],
                    "score": prev["player_penalty"],
                    "apology_decision": prev["player_apology"],
                    "apology_message": "",
                    "reason_code": ""
                })
                game.history.update_round(1, agent2_name, {
                    "strategy": prev["opponent_action"],
                    "score": prev["opponent_penalty"],
                    "apology_decision": prev["opponent_apology"],
                    "apology_message": "",
                    "reason_code": ""
                })

                # Skip round 1 execution — its data is pre-populated above.
                # The game loop starts at round 2, which is the only LLM call.
                game.current_round = 2

            games.append(game)

        return games

    # =========================================================
    # EXECUTION
    # =========================================================

    def create_games(self, config):
        experiment_cfg = config.get("experiment", {})
        mode = experiment_cfg.get("mode", "interactive")
        use_scenario_gen = experiment_cfg.get("scenarioGeneration", False)

        for lang in config["languages"]:
            config_df = self._generate_language_config_df(config, lang)

            self.config_all_langs_df = pd.concat(
                [self.config_all_langs_df, config_df],
                ignore_index=True
            )

        if mode == "controlled" and use_scenario_gen:
            self.games = []
            for _, row in self.config_all_langs_df.iterrows():
                self.games.extend(
                    self._create_controlled_experiment_games(config, row)
                )
        else:
            self.games = [
                self._create_single_game(config, row, config["payoffMatrix"])
                for _, row in self.config_all_langs_df.iterrows()
            ]

        return self.games

    def run_games(self):
        print(f"RUNNING {len(self.games)} GAMES")

        for i, game in enumerate(self.games):
            print(f"Game {i}:")

            history = game.run()

            self._upload_output(game, history, i)

            print("Completed")

    def create_and_run_games(self, config):
        processed_config = self.io_manager.process_and_validate_configuration(config)

        self.create_games(processed_config)
        self.run_games()

        return self.output_dict





# import itertools
# import pandas as pd

# from src.fairgame import FairGame
# from src.agent import Agent
# from src.io_managers.io_manager import IoManager


# class FairGameFactory:
#     """Factory class responsible for loading a configuration dictionary,
#     generating agent/game permutations, creating FairGame instances, running
#     games, and collecting output.

#     Attributes:
#         io_manager (IoManager): The I/O manager instance.
#         config_all_langs_df (pd.DataFrame): DataFrame that stores all language configuration permutations.
#         games (list): List of FairGame instances.
#         output_dict (dict): Dictionary to store game results keyed by game number.
#     """

#     def __init__(self):
#         """Initialize a new instance of FairGameFactory.

#         Initializes:
#             io_manager (IoManager): An instance of IoManager.
#             config_all_langs_df (pd.DataFrame): An empty DataFrame for configuration data.
#             games (list): An empty list to hold FairGame instances.
#             output_dict (dict): An empty dictionary to store game output.
#         """
#         self.io_manager = IoManager()
#         self.config_all_langs_df = pd.DataFrame()
#         self.games = []
#         self.output_dict = {}

#     def _generate_language_config_df(self, config, lang):
#         """Generate configuration DataFrame for a single language.

#         Depending on the configuration, either computes all agent permutations
#         or builds a single configuration.

#         Args:
#             config (dict): Configuration dictionary.
#             lang (str): Language identifier.

#         Returns:
#             pd.DataFrame: A DataFrame containing the game configurations for the specified language.
#         """
#         if config["allAgentPermutations"]:
#             return self.compute_all_game_configurations(
#                 lang,
#                 config['agents'],
#                 config['llm']
#             )
#         return self.compute_configuration(lang, config['agents'], config['llm'])

#     def _compute_agent_configurations(self, lang, config_agents):
#         """Generate all agent configurations including names, personality, and knowledge permutations.

#         Computes all combinations of agent names, personalities, and opponent personality probabilities.

#         Args:
#             lang (str): Language identifier.
#             config_agents (dict): Dictionary containing agent configurations with keys such as 'names',
#                                   'personalities', and 'opponentPersonalityProb'.

#         Returns:
#             tuple: A tuple containing:
#                 - agent_combinations (list): List of agent name combinations.
#                 - personality_permutations (list): List of personality permutations.
#                 - knowledge_permutations (list): List of knowledge permutations.
#         """
#         n_agents = len(config_agents['names'])
#         agent_combinations = [config_agents['names']]
#         personality_permutations = list(
#             itertools.product(config_agents['personalities'][lang], repeat=n_agents)
#         )
#         knowledge_permutations = list(
#             itertools.product(config_agents['opponentPersonalityProb'], repeat=n_agents)
#         )
#         return (
#             agent_combinations,
#             personality_permutations,
#             knowledge_permutations
#         )

#     def _generate_full_permutations(self, agent_combinations, personality_permutations, knowledge_permutations):
#         """Generate full configuration permutations for agents.

#         Constructs a DataFrame from all combinations of agents, personality permutations,
#         and knowledge permutations.

#         Args:
#             agent_combinations (list): List of agent combinations.
#             personality_permutations (list): List of personality tuples.
#             knowledge_permutations (list): List of knowledge tuples.

#         Returns:
#             pd.DataFrame: DataFrame containing all possible game configuration permutations.
#         """
#         rows = []
#         for agents in agent_combinations:
#             n_agents = len(agents)
#             for personality_tuple, knowledge_tuple in itertools.product(personality_permutations, knowledge_permutations):
#                 row_dict = {
#                     **{f"Agent{i+1}": agents[i] for i in range(n_agents)},
#                     **{f"Personality{i+1}": personality_tuple[i] for i in range(n_agents)},
#                     **{f"OpponentPersonalityProb{i+1}": knowledge_tuple[i] for i in range(n_agents)}
#                 }
#                 rows.append(row_dict)
#         return pd.DataFrame(rows)

#     def compute_all_game_configurations(self, lang, config_agents, llm_service):
#         """Create a DataFrame of all permutations for a given language and LLM service.

#         Args:
#             lang (str): Language identifier.
#             config_agents (dict): Agent configuration dictionary.
#             llm_service: The LLM service to be used.

#         Returns:
#             pd.DataFrame: DataFrame containing all game configurations.
#         """
#         agent_combinations, pers_perms, knowledge_perms = self._compute_agent_configurations(lang, config_agents)
#         df = self._generate_full_permutations(agent_combinations, pers_perms, knowledge_perms)
#         df['LLM'] = llm_service
#         df['Language'] = lang
#         return df

#     def compute_configuration(self, lang, config_agents, llm_service):
#         """Generate a single configuration DataFrame for the first configuration.

#         Useful when not generating all permutations.

#         Args:
#             lang (str): Language identifier.
#             config_agents (dict): Agent configuration dictionary.
#             llm_service: The LLM service to be used.

#         Returns:
#             pd.DataFrame: A single-row DataFrame with the configuration.
#         """
#         n_agents = len(config_agents['names'])
#         row_dict = {}
#         for i in range(n_agents):
#             row_dict[f"Agent{i+1}"] = config_agents['names'][i]
#             row_dict[f"Personality{i+1}"] = config_agents['personalities'][lang][i]
#             row_dict[f"OpponentPersonalityProb{i+1}"] = config_agents['opponentPersonalityProb'][i]
#         row_dict["LLM"] = llm_service
#         row_dict["Language"] = lang
#         return pd.DataFrame([row_dict])

#     def _create_single_game(
#         self,
#         config,
#         game_config_row,
#         payoff_matrix
#         ):
#         """
#         Instantiate a single FairGame
#         using both gameplay and apology prompts.
#         """

        
#         game_prompt_template = self.build_prompt_template(
#             config,
#             game_config_row['Language'],
#             stage="game"
#         )

#         apology_prompt_template = self.build_prompt_template(
#             config,
#             game_config_row['Language'],
#             stage="apology"
#         )

#         agents = self.create_agents(game_config_row)

#         return FairGame(
#             config['name'],
#             game_config_row['Language'],
#             agents,
#             config['nRounds'],
#             config['nRoundsIsKnown'],
#             payoff_matrix,

#             # NEW:
#             game_prompt_template,
#             apology_prompt_template,

#             config['stopGameWhen'],
#             config['agentsCommunicate'],

#             # Full config
#             config
#         )
    

#     def create_agents(self, game_config_row):
#         """Create agents based on the configuration row.

#         Dynamically creates Agent instances for each agent defined in the configuration.
#         Expected columns in the row include: Agent1, Personality1, OpponentPersonalityProb1, etc.

#         Args:
#             game_config_row (pd.Series): A row from the game configuration DataFrame with agent details.

#         Returns:
#             dict: A dictionary of Agent instances keyed by agent names.
#         """
#         agents_dict = {}
#         i = 1
#         while f"Agent{i}" in game_config_row:
#             agent_name = game_config_row[f"Agent{i}"]
#             personality = game_config_row[f"Personality{i}"]
#             knowledge = game_config_row[f"OpponentPersonalityProb{i}"]
#             agents_dict[agent_name] = Agent(
#                 name=agent_name,
#                 llm_service=game_config_row['LLM'],
#                 personality=personality,
#                 opponent_personality_prob=knowledge
#             )
#             i += 1
#         return agents_dict
    
#     def _upload_output(self, game, game_history, game_n):
#         """Store game result in the output dictionary.

#         Removes the payoff matrix from the game description and saves both the
#         game description and its history.

#         Args:
#             game (FairGame): The FairGame instance.
#             game_history: The history object returned from running the game.
#             game_n (int): The game number used as an incremental index.
#         """
#         # Remove payoff matrix from the output to keep it concise
#         game.description.pop('payoff_matrix', None)
#         self.output_dict[f'game_{game_n}'] = {
#             'description': game.description,
#             'history': game_history.describe()
#         }
    
#     def set_io_manager(self, manager):
#         """Set a custom IoManager instance.

#         Args:
#             manager (IoManager): A custom IoManager instance.
#         """
#         self.io_manager = manager

#     def results_games(self):
#         """Retrieve game results.

#         Returns:
#             dict: Dictionary of game results.
#         """
#         return self.output_dict

#     def all_game_configurations(self):
#         """Get all game configurations.

#         Returns:
#             pd.DataFrame: DataFrame of all game configurations across languages.
#         """
#         return self.config_all_langs_df

#     def load_config(self, config_filename):
#         """Load configuration dictionary from a file using IoManager.

#         Args:
#             config_filename (str): The filename of the configuration file.

#         Returns:
#             dict: Loaded configuration dictionary.
#         """
#         return self.io_manager.load_config(config_filename)

#     def create_games(self, config):
#         """Create FairGame instances from the configuration dictionary.

#         For each language in the configuration, generates game configurations,
#         concatenates them, and instantiates a FairGame for each configuration row.

#         Args:
#             config (dict): The processed configuration dictionary.

#         Returns:
#             list: A list of FairGame instances.
#         """
#         for lang in config['languages']:
#             config_df = self._generate_language_config_df(config, lang)
#             self.config_all_langs_df = pd.concat(
#                 [self.config_all_langs_df, config_df],
#                 ignore_index=True
#             )

#         # Create FairGame instances
#         self.games = [
#             self._create_single_game(config, row, config['payoffMatrix'])
#             for _, row in self.config_all_langs_df.iterrows()
#         ]
#         return self.games

#     def games_info(self):
#         """Get descriptions of all created games.

#         Returns:
#             list: List of textual descriptions for each FairGame.
#         """
#         return [game.description() for game in self.games]

#     def run_games(self):
#         """Execute all FairGame instances and capture their outputs.

#         Runs each game, uploads the output, and prints progress.
#         """
#         print(f"RUNNING {len(self.games)} GAMES")
#         for i, game in enumerate(self.games):
#             print(f"Game {i}:")
#             game_history = game.run()
#             self._upload_output(game, game_history, i)
#             print('Completed')

#     def load_config_create_and_run_games(self, config_filename):
#         """Convenience method to load config, create games, run them, and return results.

#         Args:
#             config_filename (str): Filename of the configuration file.

#         Returns:
#             dict: Dictionary containing results of all games.
#         """
#         config = self.load_config(config_filename)
#         return self.create_and_run_games(config)

#     def create_and_run_games(self, config):
#         """Validate configuration, create games, execute them, and return results.

#         Args:
#             config (dict): The raw configuration dictionary.

#         Returns:
#             dict: Dictionary containing the game results.
#         """
#         processed_config = self.io_manager.process_and_validate_configuration(
#             config
#         )
#         self.create_games(processed_config)
#         self.run_games()
#         return self.output_dict

#     def build_prompt_template(
#         self,
#         config,
#         lang,
#         stage="game"
#         ):
#         """
#         Retrieve or load prompt templates
#         for multi-stage FAIRGAME execution.


#         Args:
#             config (dict):
#                 Configuration dictionary.

#             lang (str):
#                 Language code.

#             stage (str):
#                 "game" or "apology"

#         Returns:
#             str:
#                 Prompt template text.
#         """

#         try:

#             if stage == "game":

#                 template = config['promptTemplate'][lang]

#             elif stage == "apology":

#                 template = config['apology'][
#                     'promptTemplate'
#                 ][lang]

#             else:

#                 raise ValueError(
#                     f"Unknown stage: {stage}"
#                 )

#         except KeyError:

#             template_filename = (
#                 config['templateFilename']
#             )

#             template = self.io_manager.load_stage_prompt(
#                 template_filename,
#                 lang,
#                 stage=stage
#             )

#         return template
        

