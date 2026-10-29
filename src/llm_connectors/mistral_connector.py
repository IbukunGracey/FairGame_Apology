
from mistralai import Mistral
import os 
import time
import random
from mistralai.models.sdkerror import SDKError

from src.llm_connectors.abstract_connector import AbstractConnector

class MistralConnector(AbstractConnector):
    """
    Chat model implementation for the Mistral API.
    """

    def __init__(self, provider_model: str):
        self.api_key = os.getenv("API_KEY_MISTRAL")
        if not self.api_key:
            raise EnvironmentError("API_KEY_MISTRAL not found in environment variables.")
        self.provider_model = provider_model
        self.client = Mistral(api_key=self.api_key)

    # def send_prompt(self, prompt: str) -> str:
    #     response = self.client.chat.complete(
    #         model=self.provider_model,
    #         messages=[{"role": "user", "content": prompt}]
    #     )
    #     return response.choices[0].message.content

    def send_prompt(self, prompt: str) -> str:
        max_retries = 20

        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.complete(
                    model=self.provider_model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

                return response.choices[0].message.content

            except SDKError as e:
                error_text = str(e).lower()

                # Retry rate-limit errors only.
                if (
                    "429" not in error_text
                    and "rate_limit" not in error_text
                    and "rate limit" not in error_text
                ):
                    raise

                if attempt == max_retries:
                    print(
                        f"Mistral rate limit persisted after "
                        f"{max_retries} retries."
                    )
                    raise

                # Exponential backoff with a small random delay.
                delay = min(2 ** attempt, 60) + random.uniform(0, 1)

                print(
                    f"Mistral rate limit reached. "
                    f"Retry {attempt + 1}/{max_retries} "
                    f"in {delay:.1f} seconds."
                )

                time.sleep(delay)

