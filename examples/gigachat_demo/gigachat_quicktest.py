from __future__ import annotations

import base64
import json
import uuid
from dataclasses import dataclass
from typing import Any

import requests

CLIENT_ID = "your-client-id"
CLIENT_SECRET = "your-client-secret"
SCOPE = "GIGACHAT_API_PERS"

AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
API_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
VERIFY_SSL = False
DEMO_QUESTION = "Что входит в возможности GigaChat Pro по сравнению с базовым GigaChat?"
DEFAULT_MODEL = "GigaChat-Pro"

def get_token(client_id: str, client_secret: str, scope: str) -> str:
    basic_auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {
        "Authorization": f"Basic {basic_auth}",
        "RqUID": str(uuid.uuid4()),
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"scope": scope}
    resp = requests.post(AUTH_URL, headers=headers, data=data, verify=VERIFY_SSL, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]

def list_models(token: str) -> list[str]:
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(f"{API_BASE_URL}/models", headers=headers, verify=VERIFY_SSL, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    return [m.get("id") for m in payload.get("data", []) if m.get("id")]

def ask(model: str, token: str, question: str) -> str:
    headers = {"Authorization": f"Bearer {token}"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "temperature": 0.2,
    }
    resp = requests.post(f"{API_BASE_URL}/chat/completions", json=body, headers=headers, verify=VERIFY_SSL, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(data, ensure_ascii=False)

@dataclass
class Settings:
    client_id: str = CLIENT_ID
    client_secret: str = CLIENT_SECRET
    scope: str = SCOPE
    model: str = DEFAULT_MODEL


def main():
    cfg = Settings()
    token = get_token(cfg.client_id, cfg.client_secret, cfg.scope)
    models = list_models(token)
    print("Доступные модели:", ", ".join(models))
    model = cfg.model if cfg.model in models else (models[0] if models else cfg.model)
    answer = ask(model, token, DEMO_QUESTION)
    print("Вопрос:", DEMO_QUESTION)
    print("Ответ:", answer)


if __name__ == "__main__":
    main()
