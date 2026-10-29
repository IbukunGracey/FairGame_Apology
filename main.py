import sys
import os
from pathlib import Path
from typing import Dict, Any

import requests
from dotenv import load_dotenv

from src.io_managers.file_manager import FileManager
from src.results_processing.results_processor import ResultsProcessor

RESOURCES_PATH = Path("resources")
TEMPLATES_PATH = RESOURCES_PATH / "game_templates"
CONFIG_PATH = RESOURCES_PATH / "config"
RESULTS_PATH = RESOURCES_PATH / "results"

HEADERS = {
"Content-Type": "application/json"
}

def load_env_variables() -> str:
    """
    Load environment variables and return the FAIRGAME API URL.
    """


    load_dotenv()

    return os.getenv(
        "FAIRGAME_URL",
        "http://127.0.0.1:5003/create_and_run_games"
    )
    

class GamesRunner:

    def __init__(
        self,
        call_type: str,
        config: Dict[str, Any],
        game_templates: Dict[str, str],
        apology_templates: Dict[str, str],
        fairgame_url: str,
        controlled_templates: Dict[str, str] = None,
        controlled_next_action_templates: Dict[str, str] = None
    ) -> None:

        self.call_type = call_type

        self.config = config

        self.config["promptTemplate"] = game_templates

        self.config["apologyPromptTemplate"] = apology_templates

        if controlled_templates:
            self.config["controlledPromptTemplate"] = controlled_templates

        if controlled_next_action_templates:
            self.config["controlledNextActionPromptTemplate"] = (
                controlled_next_action_templates
            )

        self.fairgame_url = fairgame_url

    def run(self) -> Dict[str, Any]:

        if self.call_type == "local":
            return self._local_call()

        if self.call_type == "api":
            return self._api_call()

        raise ValueError(
            "Invalid call type. Expected 'local' or 'api'."
        )

    def _local_call(self) -> Dict[str, Any]:

        from src.fairgame_factory import FairGameFactory

        game_factory = FairGameFactory()

        return game_factory.create_and_run_games(
            self.config
        )

    def _api_call(self) -> Dict[str, Any]:

        response = requests.post(
            self.fairgame_url,
            json=self.config,
            headers=HEADERS
        )

        response.raise_for_status()

        return response.json()
    

def parse_call_type(argv: list) -> str:
    """
    Extract call type from CLI arguments.
    """

    
    if len(argv) < 2:
        raise ValueError(
            "Call type argument ('local' or 'api') is required."
        )

    return argv[1]
    

def load_template_file(
template_name: str,
language: str
) -> str:
    """
    Load a template file by name and language.
    """

    
    template_filepath = (
        TEMPLATES_PATH
        / f"{template_name}_{language}.txt"
    )

    return FileManager.read_template_file(
        template_filepath
    )
    

def load_config_file(
config_dir: str,
config_name: str
) -> Dict[str, Any]:
    """
    Load a FAIRGAME configuration JSON file.
    """

    config_filepath = (
        CONFIG_PATH
        / config_dir
        / f"{config_name}.json"
    )

    return FileManager.read_json_file(
        config_filepath
    )
    

def save_results(
results: Dict[str, Any],
config_name: str
) -> None:
    """
    Convert results into DataFrame and save CSV.
    """

    results_processor = ResultsProcessor()

    df = results_processor.process(results)

    results_filepath = (
        RESULTS_PATH
        / f"results_{config_name}.csv"
    )

    FileManager.save_results_csv(
        df,
        results_filepath
    )
    

def main() -> None:
    """
    Main FAIRGAME execution entrypoint.
    """

    # ==========================================
    # CALL TYPE
    # ==========================================

    call_type = parse_call_type(sys.argv)

    # ==========================================
    # ENVIRONMENT
    # ==========================================

    fairgame_url = load_env_variables()

    # ==========================================
    # CONFIGURATION
    # ==========================================

    config_dir = "prisoner_dilemma"

    #change the config name to prisoner_dilemma_apology to run the interactive experiment
    config_name = "prisoner_dilemma_controlled_scenarios"

    game_template_name = "prisoner_dilemma"

    apology_template_name = (
        "prisoner_dilemma_apology"
    )

    language = "en"

    # ==========================================
    # LOAD FILES
    # ==========================================

    config = load_config_file(
        config_dir,
        config_name
    )

    game_template = load_template_file(
        game_template_name,
        language
    )

    apology_template = load_template_file(
        apology_template_name,
        language
    )

    controlled_templates = None
    controlled_next_action_templates = None
    experiment_mode = config.get(
        "experiment", {}
    ).get("mode", "interactive")

    if experiment_mode == "controlled":
        controlled_template_name = (
            game_template_name + "_controlled"
        )
        controlled_template = load_template_file(
            controlled_template_name,
            language
        )
        controlled_templates = {
            language: controlled_template
        }

        controlled_na_template_name = (
            game_template_name
            + "_controlled_next_action"
        )
        controlled_na_template = load_template_file(
            controlled_na_template_name,
            language
        )
        controlled_next_action_templates = {
            language: controlled_na_template
        }

    # ==========================================
    # RUN GAMES
    # ==========================================
    for run_index in range(1):
        print(f"\n--- Starting run {run_index + 1}/1 ---")
        runner = GamesRunner(
            call_type=call_type,
            config=config,
            game_templates={
                language: game_template
            },
            apology_templates={
                language: apology_template
            },
            fairgame_url=fairgame_url,
            controlled_templates=controlled_templates,
            controlled_next_action_templates=controlled_next_action_templates
        )

        results = runner.run()

    # ==========================================
    # SAVE RESULTS
    # ==========================================

    # save_results(
    #     results,
    #     config_name
    # )
    # 6. Save each run separately
        save_results(results, f"{config_name}_{run_index}")

        print(f"--- Finished run {run_index + 1}/1 ---")

if __name__ == "__main__":
    main()

