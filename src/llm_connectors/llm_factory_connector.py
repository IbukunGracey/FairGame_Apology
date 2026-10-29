
from dotenv import load_dotenv
from src.llm_connectors.anthropic_connector import AnthropicConnector
from src.llm_connectors.mistral_connector import MistralConnector
from src.llm_connectors.openai_connector import OpenAIConnector
from src.llm_connectors.openrouter_connector import OpenRouterConnector

# Load environment variables from a .env file
load_dotenv()

# Dictionary mapping our abstract model names to provider classes and their corresponding provider model identifiers.
MODEL_PROVIDER_MAP = {
    "Claude35Sonnet": (AnthropicConnector, "claude-3-5-sonnet-20241022"),
    "MistralLarge": (MistralConnector, "mistral-large-latest"),
    "MistralLarge3": (MistralConnector, "mistral-large-2512"),
    "OpenAIGPT4o": (OpenAIConnector, "gpt-4o"),
    # OpenRouter models — any model available on OpenRouter can be added here.
    "OR_Claude35Sonnet": (OpenRouterConnector, "anthropic/claude-3.5-sonnet"),
    "OR_Claude48": (OpenRouterConnector, "anthropic/claude-opus-4.8"),
    "OR_GPT4o": (OpenRouterConnector, "openai/gpt-4o"),
    "OR_Llama370B": (OpenRouterConnector, "meta-llama/llama-3-70b-instruct"),
    "OR_Llama4Scout": (OpenRouterConnector, "meta-llama/llama-4-scout"),
    "OR_Gemini15Pro": (OpenRouterConnector, "google/gemini-pro-1.5"),
    "OR_Gemini31Pro": (OpenRouterConnector, "google/gemini-3.1-pro-preview"),
    "OR_MistralLarge": (OpenRouterConnector, "mistralai/mistral-large"),
    "OR_MistralLarge3": (OpenRouterConnector, "mistralai/mistral-large-2512"),
    "OR_GPT5.5": (OpenRouterConnector, "openai/gpt-5.5"),
}

class ChatModelFactory:
    """
    Factory for creating chat model instances based on the model name.
    """

    @staticmethod
    def get_model(model_name: str):
        """
        Return an instance of ChatModel based on the provided model name.

        Parameters:
            model_name (str): The abstract model name (e.g., "MistralLarge", "GPT4", "Claude").

        Returns:
            ChatModel: An instance of the appropriate chat model.

        Raises:
            ValueError: If the model_name is unsupported.
        """
        provider_info = MODEL_PROVIDER_MAP.get(model_name)
        if not provider_info:
            raise ValueError(f"Unsupported model specified: {model_name}")
        model_class, provider_model = provider_info
        return model_class(provider_model)


def execute_prompt(model_name: str, prompt: str) -> str:
    """
    Execute a prompt using the specified model.

    Parameters:
        model_name (str): The abstract model name (e.g., "MistralLarge", "GPT4", "Claude").
        prompt (str): The prompt text to send to the API.

    Returns:
        str: The response text from the API.
    """
    chat_model = ChatModelFactory.get_model(model_name)
    return chat_model.send_prompt(prompt)


if __name__ == "__main__":
    # Example usage:
    model_identifier = "Claude35Sonnet"
    prompt_text = "Tell me a programming joke."
    
    try:
        result = execute_prompt(model_identifier, prompt_text)
        print("API Response:", result)
    except Exception as e:
        print("An error occurred:", e)
