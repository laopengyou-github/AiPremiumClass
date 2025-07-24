import requests
import json
import torch

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    url = 'http://localhost:11434/api/chat'
    data = {
        "model" : "qwen3",
        "messages": [
            {"role": "user", "content": "天为什么是红的"}
        ],
        "stream" : False
    }
    response = requests.post(url, json = data)
    print(response.json())
    print(response.json()['message']['content'])