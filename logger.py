import logging
import sys

# 配置日志器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 文件处理器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler = logging.FileHandler('livetalking.log')
fhandler.setFormatter(formatter)
fhandler.setLevel(logging.INFO)
logger.addHandler(fhandler)

# 控制台处理器
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
sformatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(sformatter)
logger.addHandler(handler)