from openai import OpenAI
import os

from src.llm_connectors.abstract_connector import AbstractConnector


class OpenRouterConnector(AbstractConnector):
    """
    Chat model implementation for the OpenRouter API.
    Uses the OpenAI-compatible endpoint at https://openrouter.ai/api/v1.
    """

    def __init__(self, provider_model: str, temperature: float = 1.0):
        self.api_key = os.getenv("API_KEY_OPENROUTER")
        if not self.api_key:
            raise EnvironmentError("API_KEY_OPENROUTER not found in environment variables.")
        self.provider_model = provider_model
        self.temperature = temperature
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    # def send_prompt(self, prompt: str) -> str:
    #     messages = [{"role": "user", "content": prompt}]
    #     completion = self.client.chat.completions.create(
    #         model=self.provider_model,
    #         temperature=self.temperature,
    #         messages=messages,
    #     )
    #     return completion.choices[0].message.content

    
    def send_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]

        completion = self.client.chat.completions.create(
            model=self.provider_model,
            temperature=self.temperature,
            messages=messages,
        )

        # Check the response structure.
        if not completion.choices:
            raise RuntimeError(
                f"OpenRouter returned no choices for "
                f"{self.provider_model!r}. "
                f"Response ID: {completion.id!r}"
            )

        choice = completion.choices[0]
        message = choice.message
        content = message.content

        # Log diagnostic information when the response has no text.
        if content is None or not content.strip():
            usage = completion.usage

            print("\n=== EMPTY OPENROUTER RESPONSE ===")
            print("Requested model:", self.provider_model)
            print("Returned model:", completion.model)
            print("Response ID:", completion.id)
            print("Finish reason:", choice.finish_reason)
            print(
                "Native finish reason:",
                getattr(choice, "native_finish_reason", None)
            )
            print("Message:", repr(message))
            print("Usage:", repr(usage))
            print(
                "Choice error:",
                repr(getattr(choice, "error", None))
            )
            print("=================================\n")

            raise RuntimeError(
                f"OpenRouter returned no text for "
                f"{self.provider_model!r}. "
                f"Finish reason: {choice.finish_reason!r}; "
                f"response ID: {completion.id!r}. "
                "See diagnostic output above."
            )

        return content


        
