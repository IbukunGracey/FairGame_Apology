import os
from pathlib import Path

from src.io_managers.file_manager import FileManager
from src.io_managers.configuration_validator import ConfigValidator
from src.utils.utils import get_project_root

# script_path = os.path.abspath(__file__)

script_path = Path(__file__).resolve()
project_root = get_project_root(script_path, 3)

DEFAULT_RESOURCES = project_root / "resources"

class IoManager:
    """
    Orchestrates reading config files, validating structure,
    and loading prompt templates for multi-stage FAIRGAME games.
    """


    def __init__(
        self,
        root_path: str = DEFAULT_RESOURCES,
        config_path: str = "config",
        game_path: str = "game_templates",
    ):
        self.file_manager = FileManager()
        self.config_validator = ConfigValidator()

        self.config_path = Path(root_path) / config_path
        self.game_path = Path(root_path) / game_path

    def load_config(self, config_filename: str) -> dict:
        """
        Loads a configuration file (JSON) from disk.
        """
        config_filepath = self.config_path / config_filename
        return self.file_manager.read_json_file(config_filepath)

    def process_and_validate_configuration(self, config_data: dict) -> dict:
        """
        Validates configuration structure and integrity.
        """
        return self.config_validator.validate_config_structure(config_data)

    def load_template(self, filename: str, lang: str) -> str:
        """
        Loads a game template file based on filename and language.
        Used for Stage 1 (gameplay prompt).
        """
        template_filepath = self.game_path / f"{filename}_{lang}.txt"
        return self.file_manager.load_text_file(template_filepath)

    def load_stage_prompt(self, filename: str, lang: str, stage: str = "game") -> str:
        """
        Loads prompt templates for multi-stage gameplay.

        Args:
            filename: base filename (e.g. 'prisoner_dilemma')
            lang: language code (e.g. 'en')
            stage: 'game' or 'apology'

        Returns:
            Prompt template text
        """

        if stage == "game":
            filepath = self.game_path / f"{filename}_{lang}.txt"

        elif stage == "apology":
            filepath = self.game_path / f"{filename}_apology_{lang}.txt"

        elif stage == "controlled":
            filepath = self.game_path / f"{filename}_controlled_{lang}.txt"

        elif stage == "controlled_next_action":
            filepath = self.game_path / f"{filename}_controlled_next_action_{lang}.txt"

        else:
            raise ValueError(f"Unknown stage: {stage}")

        return self.file_manager.load_text_file(filepath)
