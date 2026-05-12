import time
import requests
from basereal import BaseReal
from logger import logger

def llm_response(message,nerfreal:BaseReal):
    start = time.perf_counter()
    from openai import OpenAI
    client = OpenAI(
        # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        api_key="61d43b22099a4692930cb67c48f066e7.7CXkwBJdKTMguQew",
        # 填写DashScope SDK的base_url
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )
    end = time.perf_counter()
    logger.info(f"llm Time init: {end-start}s")
    completion = client.chat.completions.create(
        model="glm-4.7",
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                  {'role': 'user', 'content': message}],
        stream=True,
        # 通过以下设置，在流式输出的最后一行展示token使用信息
        stream_options={"include_usage": True}
    )
    result=""
    first = True
    for chunk in completion:
        if len(chunk.choices)>0:
            #print(chunk.choices[0].delta.content)
            if first:
                end = time.perf_counter()
                logger.info(f"llm Time to first chunk: {end-start}s")
                first = False
            msg = chunk.choices[0].delta.content
            if msg is not None:
                lastpos=0
                #msglist = re.split('[,.!;:，。！?]',msg)
                for i, char in enumerate(msg):
                    if char in ",.!;:，。！？：；" :
                        result = result+msg[lastpos:i+1]
                        lastpos = i+1
                        # 检查是否只有标点符号
                        if result.strip(' ,.!;:，。！？：；') and len(result)>10:
                            logger.info(result)
                            # 确保内容是 UTF-8 格式
                            result_utf8 = result.encode('utf-8').decode('utf-8')
                            nerfreal.put_msg_txt(result_utf8)
                            result=""
                result = result+msg[lastpos:]
    end = time.perf_counter()
    logger.info(f"llm Time to last chunk: {end-start}s")
    # 检查是否只有标点符号
    if result.strip(' ,.!;:，。！？：；'):
        # 确保内容是 UTF-8 格式
        result_utf8 = result.encode('utf-8').decode('utf-8')
        nerfreal.put_msg_txt(result_utf8)

def ai_agent_response(message,nerfreal:BaseReal):
    # ──────────────────────────────────────────────
    # 智能体 API 配置（填写你的接口信息）
    # ──────────────────────────────────────────────
    AI_AGENT_URL   = "https://your-agent-api.com/v1/chat"   # TODO: 替换为实际 API 地址
    AI_AGENT_TOKEN = "your-api-token-here"                   # TODO: 替换为实际 Token/Key
    # 请求体字段名：发送用户消息的字段（如 "input" / "message" / "query" 等）
    AI_AGENT_INPUT_FIELD = "input"                           # TODO: 根据 API 文档调整
    # 响应体字段名：SSE data JSON 中携带文本的字段（如 "response" / "output" / "answer" 等）
    AI_AGENT_OUTPUT_FIELD = "response"                       # TODO: 根据 API 文档调整
    # 其他固定参数（如 agent_id、session_id 等，不需要的删掉）
    AI_AGENT_EXTRA_PAYLOAD = {
        # "agent_id": "your-agent-id",  # TODO: 如需要，填写 agent_id
        # "session_id": "",             # TODO: 如需要，填写 session_id
    }
    # ──────────────────────────────────────────────

    start = time.perf_counter()
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AI_AGENT_TOKEN}"
        }
        payload = {
            AI_AGENT_INPUT_FIELD: message,
            "stream": True,
            **AI_AGENT_EXTRA_PAYLOAD
        }
        resp = requests.post(AI_AGENT_URL, json=payload, headers=headers, stream=True)
        resp.raise_for_status()
        end = time.perf_counter()
        logger.info(f"ai_agent Time init: {end-start}s")
        
        result = ""
        first = True
        for chunk in resp.iter_content(chunk_size=1024):
            if chunk:
                try:
                    content = chunk.decode('utf-8', errors='ignore')
                    # 分割SSE消息
                    messages = content.split('\n\n')
                    for msg in messages:
                        if not msg:
                            continue
                        # 提取data字段
                        lines = msg.split('\n')
                        for line in lines:
                            if line.startswith('data:'):
                                data_str = line[5:].strip()
                                if data_str:
                                    import json
                                    try:
                                        data = json.loads(data_str)
                                        if AI_AGENT_OUTPUT_FIELD in data:
                                            msg = data[AI_AGENT_OUTPUT_FIELD]
                                            if msg is not None:
                                                if first:
                                                    end = time.perf_counter()
                                                    logger.info(f"ai_agent Time to first chunk: {end-start}s")
                                                    first = False
                                                lastpos=0
                                                for i, char in enumerate(msg):
                                                    if char in ",.!;:，。！？：；" :
                                                        result = result+msg[lastpos:i+1]
                                                        lastpos = i+1
                                                        # 检查是否只有标点符号
                                                        if result.strip(' ,.!;:，。！？：；') and len(result)>10:
                                                            logger.info(result)
                                                            # 确保内容是 UTF-8 格式
                                                            result_utf8 = result.encode('utf-8').decode('utf-8')
                                                            nerfreal.put_msg_txt(result_utf8)
                                                            result=""
                                                result = result+msg[lastpos:]
                                    except json.JSONDecodeError:
                                        pass
                except UnicodeDecodeError:
                    pass
        
        end = time.perf_counter()
        logger.info(f"ai_agent Time to last chunk: {end-start}s")
        # 发送剩余的消息
        if result.strip(' ,.!;:，。！？：；'):
            # 确保内容是 UTF-8 格式
            result_utf8 = result.encode('utf-8').decode('utf-8')
            nerfreal.put_msg_txt(result_utf8)

    except requests.exceptions.RequestException as e:
        logger.error(f"请求出错：{e}")
   