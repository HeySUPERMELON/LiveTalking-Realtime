###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

import asyncio
import json
import os
import random
from typing import Dict

from aiohttp import web
import aiohttp_cors

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.contrib.media import MediaPlayer, MediaRelay
from aiortc.rtcrtpsender import RTCRtpSender

#from webrtc import HumanPlayer
from webrtc import HumanPlayer

from basereal import BaseReal

import threading

#import multiprocessing
import torch.multiprocessing as mp


import threading

nerfreals:Dict[int, BaseReal] = {} #sessionid:BaseReal
nerfreals_lock = threading.Lock() # 线程安全锁
opt = None
model = None
avatar = None
        

#####webrtc###############################
pcs = set()

def randN(N)->int:
    '''生成长度为 N的随机数 '''
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)

def build_nerfreal(sessionid:int)->BaseReal:
    opt.sessionid=sessionid
    if opt.model == 'wav2lip':
        from lipreal import LipReal
        nerfreal = LipReal(opt,model,avatar)
    elif opt.model == 'musetalk':
        from musereal import MuseReal
        nerfreal = MuseReal(opt,model,avatar)
    # elif opt.model == 'ernerf':
    #     from nerfreal import NeRFReal
    #     nerfreal = NeRFReal(opt,model,avatar)
    elif opt.model == 'ultralight':
        from lightreal import LightReal
        nerfreal = LightReal(opt,model,avatar)
    return nerfreal

#@app.route('/offer', methods=['POST'])
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # if len(nerfreals) >= opt.max_session:
    #     logger.info('reach max session')
    #     return web.Response(
    #         content_type="application/json",
    #         text=json.dumps(
    #             {"code": -1, "msg": "reach max session"}
    #         ),
    #     )
    sessionid = randN(6) #len(nerfreals)
    with nerfreals_lock:
        nerfreals[sessionid] = None
        logger.info('sessionid=%d, session num=%d',sessionid,len(nerfreals))
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
    with nerfreals_lock:
        nerfreals[sessionid] = nerfreal
    
    #ice_server = RTCIceServer(urls='stun:stun.l.google.com:19302')
    ice_server = RTCIceServer(urls='stun:stun.freeswitch.org:3478')
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[ice_server]))
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            with nerfreals_lock:
                if sessionid in nerfreals:
                    del nerfreals[sessionid]
        if pc.connectionState == "closed":
            pcs.discard(pc)
            with nerfreals_lock:
                if sessionid in nerfreals:
                    del nerfreals[sessionid]
            # gc.collect()

    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)
    capabilities = RTCRtpSender.getCapabilities("video")
    preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
    transceiver = pc.getTransceivers()[1]
    transceiver.setCodecPreferences(preferences)

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    #return jsonify({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid":sessionid}
        ),
    )

async def human(request):
    try:
        params = await request.json()

        sessionid = params.get('sessionid',0)
        logger.info(f"Received message from session {sessionid}: type={params.get('type')}, text={params.get('text', '')[:50]}...")
        
        with nerfreals_lock:
            if sessionid not in nerfreals:
                logger.warning(f"Session {sessionid} not found")
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {"code": -1, "msg": f"Session {sessionid} not found"}
                    ),
                )
            if params.get('interrupt'):
                logger.info(f"Interrupting talk for session {sessionid}")
                nerfreals[sessionid].flush_talk()

            if params['type']=='echo':
                logger.info(f"Processing echo message for session {sessionid}")
                nerfreals[sessionid].put_msg_txt(params['text'])
            elif params['type']=='chat':
                logger.info(f"Processing chat message for session {sessionid}, sending to LLM")
                asyncio.get_event_loop().run_in_executor(None, llm_response, params['text'],nerfreals[sessionid])                         
                #nerfreals[sessionid].put_msg_txt(res)

        logger.info(f"Message processed successfully for session {sessionid}")
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def interrupt_talk(request):
    try:
        params = await request.json()

        sessionid = params.get('sessionid',0)
        with nerfreals_lock:
            if sessionid not in nerfreals:
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {"code": -1, "msg": f"Session {sessionid} not found"}
                    ),
                )
            nerfreals[sessionid].flush_talk()
        
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def humanaudio(request):
    try:
        form= await request.post()
        sessionid = int(form.get('sessionid',0))
        with nerfreals_lock:
            if sessionid not in nerfreals:
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {"code": -1, "msg": f"Session {sessionid} not found"}
                    ),
                )
            fileobj = form["file"]
            filename=fileobj.filename
            filebytes=fileobj.file.read()
            nerfreals[sessionid].put_audio_file(filebytes)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def set_audiotype(request):
    try:
        params = await request.json()

        sessionid = params.get('sessionid',0)    
        with nerfreals_lock:
            nerfreals[sessionid].set_custom_state(params['audiotype'],params['reinit'])

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def record(request):
    try:
        params = await request.json()
        sessionid = params.get('sessionid', 0)
        # 检查sessionid是否存在
        with nerfreals_lock:
            if sessionid not in nerfreals:
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {"code": -1, "msg": f"Session {sessionid} not found"}
                    ),
                )            
            if params['type']=='start_record':
                nerfreals[sessionid].start_recording()
            elif params['type']=='end_record':
                nerfreals[sessionid].stop_recording()
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def is_speaking(request):
    try:
        params = await request.json()

        sessionid = params.get('sessionid',0)
        with nerfreals_lock:
            if sessionid not in nerfreals:
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {"code": -1, "msg": f"Session {sessionid} not found"}
                    ),
                )
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": 0, "data": nerfreals[sessionid].is_speaking()}
                ),
            )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

from llm import llm_response
from logger import logger

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--avatar_id', type=str, default='avator_1')
    parser.add_argument('--fps', type=int, default=50)
    #parser.add_argument('--asr', type=str, default='funasr')
    #parser.add_argument('--asr_model', type=str, default='iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
    #parser.add_argument('--asr_model', type=str, default='iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
    #parser.add_argument('--asr_model', type=str, default='iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch-1280')
    #parser.add_argument('--asr_model', type=str, default='iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
    parser.add_argument('--asr_model', type=str, default='iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch-onnx')
    #parser.add_argument('--asr_model', type=str, default='iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch-onnx')
    #parser.add_argument('--asr_model', type=str, default='iic/speech_paraformer-tiny-commandword_asr_nat-zh-cn-16k-common-vocab5447-pytorch')
    #parser.add_argument('--asr_model', type=str, default='iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1')
    parser.add_argument('--asr_lang', type=str, default='zh')
    parser.add_argument('--asr_hotword', type=str, default='')
    parser.add_argument('--asr_access_token', type=str, default='')
    parser.add_argument('--tts', type=str, default='edgetts', help="tts service type") #xtts gpt-sovits cosyvoice fishtts tencent doubao indextts2 azuretts
    parser.add_argument('--REF_FILE', type=str, default="zh-CN-YunxiaNeural",help="参考文件名或语音模型ID，默认值为 edgetts的语音模型ID zh-CN-YunxiaNeural, 若--tts指定为azuretts, 可以使用Azure语音模型ID, 如zh-CN-XiaoxiaoMultilingualNeural")
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880') # http://localhost:9000
    # parser.add_argument('--CHARACTER', type=str, default='test')
    # parser.add_argument('--EMOTION', type=str, default='default')

    parser.add_argument('--model', type=str, default='musetalk') #musetalk wav2lip ultralight

    parser.add_argument('--transport', type=str, default='rtcpush') #webrtc rtcpush virtualcam
    parser.add_argument('--push_url', type=str, default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream') #rtmp://localhost/live/livestream

    parser.add_argument('--max_session', type=int, default=1)  #multi session count
    parser.add_argument('--listenport', type=int, default=8010, help="web listen port")

    opt = parser.parse_args()
    #app.config.from_object(opt)
    #print(app.config)
    opt.customopt = []
    if opt.customvideo_config!='':
        with open(opt.customvideo_config,'r') as file:
            opt.customopt = json.load(file)

    # if opt.model == 'ernerf':       
    #     from nerfreal import NeRFReal,load_model,load_avatar
    #     model = load_model(opt)
    #     avatar = load_avatar(opt) 
    if opt.model == 'musetalk':
        from musereal import MuseReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model()
        avatar = load_avatar(opt.avatar_id) 
        warm_up(opt.batch_size,model)      
    elif opt.model == 'wav2lip':
        from lipreal import LipReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model("./models/wav2lip.pth")
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,model,256)
    elif opt.model == 'ultralight':
        from lightreal import LightReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model(opt)
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,avatar,160)

    # if opt.transport=='rtmp':
    #     thread_quit = Event()
    #     nerfreals[0] = build_nerfreal(0)
    #     rendthrd = Thread(target=nerfreals[0].render,args=(thread_quit,))
    #     rendthrd.start()
    if opt.transport=='virtualcam':
        thread_quit = Event()
        nerfreals[0] = build_nerfreal(0)
        rendthrd = Thread(target=nerfreals[0].render,args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    appasync = web.Application(client_max_size=1024**2*100)
    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/human", human)
    appasync.router.add_post("/humanaudio", humanaudio)
    appasync.router.add_post("/set_audiotype", set_audiotype)
    appasync.router.add_post("/record", record)
    appasync.router.add_post("/interrupt_talk", interrupt_talk)
    appasync.router.add_post("/is_speaking", is_speaking)
    appasync.router.add_static('/',path='web')
    
    # 添加WebSocket路由
    async def websocket_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        logger.info('WebSocket connection established')
        
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        logger.info(f'Received WebSocket message: {msg.data[:50]}...')
                        # 解析消息
                        data = json.loads(msg.data)
                        sessionid = data.get('sessionid', 0)
                        msg_type = data.get('type', '')
                        text = data.get('text', '')
                        interrupt = data.get('interrupt', False)
                        
                        logger.info(f'Parsed message: sessionid={sessionid}, type={msg_type}, text={text[:30] if text else ""}..., interrupt={interrupt}')
                        
                        # 检查sessionid是否存在
                        with nerfreals_lock:
                            logger.info(f'Checking sessionid {sessionid} in nerfreals, available sessions: {list(nerfreals.keys())}')
                            if sessionid not in nerfreals:
                                logger.warning(f'Session {sessionid} not found in nerfreals')
                                response = {
                                    "code": -1,
                                    "msg": f"Session {sessionid} not found"
                                }
                                if not ws.closed:
                                    await ws.send_str(json.dumps(response))
                                continue
                            
                            logger.info(f'Session {sessionid} found, nerfreals[{sessionid}]={nerfreals[sessionid]}')
                            
                            # 处理中断
                            if interrupt:
                                logger.info(f"Interrupting talk for session {sessionid}")
                                nerfreals[sessionid].flush_talk()
                            
                            # 处理消息类型
                            if msg_type == 'echo':
                                logger.info(f"Processing echo message for session {sessionid}, text={text[:50]}...")
                                nerfreals[sessionid].put_msg_txt(text)
                                logger.info(f"put_msg_txt called for session {sessionid}")
                            elif msg_type == 'chat':
                                logger.info(f"Processing chat message for session {sessionid}, sending to LLM")
                                # 在后台线程中处理LLM响应
                                threading.Thread(target=llm_response, args=(text, nerfreals[sessionid])).start()
                        
                        # 返回成功响应
                        response = {
                            "code": 0,
                            "msg": "ok"
                        }
                        if not ws.closed:
                            await ws.send_str(json.dumps(response))
                        logger.info(f"Message processed successfully for session {sessionid}")
                    except json.JSONDecodeError as e:
                        logger.error(f'JSON decode error: {e}')
                        if not ws.closed:
                            response = {"code": -1, "msg": f"Invalid JSON: {str(e)}"}
                            await ws.send_str(json.dumps(response))
                    except Exception as e:
                        logger.exception('WebSocket message processing exception:')
                        if not ws.closed:
                            response = {"code": -1, "msg": str(e)}
                            await ws.send_str(json.dumps(response))
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
                elif msg.type == web.WSMsgType.CLOSED:
                    logger.info('WebSocket connection closed by client')
                    break
        except Exception as e:
            logger.exception('WebSocket handler exception:')
        finally:
            logger.info('WebSocket connection closed')
            if not ws.closed:
                await ws.close()
        
        return ws
    
    appasync.router.add_get('/humanecho', websocket_handler)

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    # Configure CORS on all routes.
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename='webrtcapi.html'
    if opt.transport=='rtmp':
        pagename='echoapi.html'
    elif opt.transport=='rtcpush':
        pagename='rtcpushapi.html'
    logger.info('start http server; http://<serverip>:'+str(opt.listenport)+'/'+pagename)
    logger.info('如果使用webrtc，推荐访问webrtc集成前端: http://<serverip>:'+str(opt.listenport)+'/dashboard.html')
    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())
        if opt.transport=='rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k!=0:
                    push_url = opt.push_url+str(k)
                loop.run_until_complete(run(push_url,k))
        loop.run_forever()    
    #Thread(target=run_server, args=(web.AppRunner(appasync),)).start()
    # 启动aiohttp服务器
    def start_aiohttp_server():
        run_server(web.AppRunner(appasync))

    # 启动aiohttp服务器（主线程）
    start_aiohttp_server()
    
    
