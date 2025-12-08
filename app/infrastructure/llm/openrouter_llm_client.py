import requests

from app.infrastructure.config import OPENROUTER_API_KEY, OPENROUTER_MODEL_NAME


class OpenRouterLlmClient:

    def __init__(
        self,
        timeout: float = 60.0,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
    ) -> None:
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        if not OPENROUTER_MODEL_NAME:
            raise RuntimeError("OPENROUTER_MODEL_NAME is not set")

        self._timeout = timeout
        self._base_url = base_url
        self._api_key = OPENROUTER_API_KEY
        self._model = OPENROUTER_MODEL_NAME

    def generate(self, prompt: str) -> str:
        response = requests.post(
            self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Ты - помощник абитуриентов и студентов университета МИРЭА. "
                            "Отвечай коротко (2-6 предложений)б простым языком. "
                            "Опирайся только на предоставленный контекст. "
                            "Если в контексте нет нужно информации, честно, скажи, "
                            "что ответить не можешь, так как не обладаешь этой информацией."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "temperature": 0.1,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
