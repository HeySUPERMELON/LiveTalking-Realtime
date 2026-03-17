from calendar import c
from nturl2path import url2pathname
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
                        if len(result)>10:
                            logger.info(result)
                            nerfreal.put_msg_txt(result)
                            result=""
                result = result+msg[lastpos:]
    end = time.perf_counter()
    logger.info(f"llm Time to last chunk: {end-start}s")
    nerfreal.put_msg_txt(result)    

def ai_agent_response(message,nerfreal:BaseReal):
    start = time.perf_counter()
    try:
        url = "https://maas-api.ai-yuanjing.com/openapi/v3/assistant/chat/completions"
        headers = {
            "Content-Type": "text/event-stream",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjM4ODU1NjQxOTAwNTMwMDg2IiwidXNlclR5cGUiOjAsInVzZXJuYW1lIjoieHVnNTIiLCJuaWNrbmFtZSI6IuiuuOingiIsImJ1ZmZlclRpbWUiOjE3NzM3MjIwMDksImNyZWF0ZWRCeSI6IjM4ODU1NjQxOTAwNTMwMDg2IiwidGVuYW50SUQiOiIzODg1NTY0MTkwMDUzMDA4NiIsInN1YlVzZXJuYW1lIjoieHVnNTIiLCJzdWJOaWNrbmFtZSI6IuiuuOingiIsImV4cCI6MTc3Mzc3MjQwOSwianRpIjoiZWIyOTIwMmItMGYxZS00ODc2LWE4ZDUtM2I5ZmU4N2E2NGU3IiwiaWF0IjoxNzczNzE0Njg5LCJpc3MiOiIzODg1NTY0MTkwMDUzMDA4NiIsIm5iZiI6MTc3MzcxNDY4OSwic3ViIjoid2ViIn0.OPjpdfod3-L6n1WJc4yN3de9m_-RSDfQUkiuabx8OD8"
        }
        payload = {
            "agent_id": "5e64509e-6ec4-4c1c-bed6-827e3849333e",
            "session_id": "",
            "input": message,
            "stream": True,
            "upload_file_url": ""
        }
        resp = requests.post(url, json=payload, headers=headers, stream=True)
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
                                        if 'response' in data:
                                            msg = data['response']
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
                                                        nerfreal.put_msg_txt(result)
                                                        result=""
                                                result = result+msg[lastpos:]
                                    except json.JSONDecodeError:
                                        pass
                except UnicodeDecodeError:
                    pass
        
        end = time.perf_counter()
        logger.info(f"ai_agent Time to last chunk: {end-start}s")
        # 发送剩余的消息
        if result:
            nerfreal.put_msg_txt(result)

    except requests.exceptions.RequestException as e:
        logger.error(f"请求出错：{e}")
