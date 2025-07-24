import requests
import json

if __name__ == '__main__':
    url = "http://localhost:11434/api/generate"

    data = {
        "model": "deepseek-r1:1.5b",
        "prompt": "天空是什么颜色的",
        "stream": False
    }

    resp = requests.post(url, json=data)

    if resp.status_code == 200:
        print(json.loads(resp.text)['response'])