# deepseek_adapter.py
import requests
from utils import deepseek_api_key

def chat_completion(messages, model="default", temperature=0.7, max_tokens=512):
    url = "https://api.deepseek.ai/v1/completions"
    headers = {
        "Authorization": f"Bearer {deepseek_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    # 假設 Deepseek 回傳結構同 OpenAI 類似
    return data["choices"][0]["message"]["content"]
