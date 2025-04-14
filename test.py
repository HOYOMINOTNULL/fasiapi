
import requests
import json

# 发送 POST 请求，启用流式处理
response = requests.post(
    "http://9f62bd1.r9.cpolar.top/api/generate",
    json={
        "model": "myllama",
        "prompt": "为什么工地上要戴安全帽？"
    },
    stream=True  # 开启流式响应
)

# 逐行处理响应
for line in response.iter_lines():
    if line:  # 确保行不为空
        try:
            # 将每行解码并解析为 JSON 对象
            json_data = json.loads(line.decode('utf-8'))
            # 如果 JSON 包含 "response" 字段，打印出来
            if "response" in json_data:
                print(json_data["response"], end='')  # 不换行，拼接输出
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}")

