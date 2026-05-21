import numpy as np
import cv2
import copy

# ──────────────────────────────────────────────────────────────────────
# Mask 混合可配置参数 — 控制嘴形可见幅度与自然度
# ──────────────────────────────────────────────────────────────────────

# 全局 Gamma 校正系数 — 整体提升 mask 对比度
#   1.0 = 原始（无修改），0.5 = 大幅增强嘴形，0.3 = 极度增强（可能有接缝）
#   原理：mask 边缘值 0.5 → gamma(0.5,0.5)=0.71 → 生成的嘴占比从 50% 提升到 71%
MOUTH_MASK_GAMMA = 0.6

# 嘴部区域定向增强 — 核心嘴区（mask 高值区域）额外增加权重
#   0.0 = 关闭（仅用全局 gamma），0.1-0.2 = 温和增强，0.3+ = 强力增强
#   原理：mask 值 > MOUTH_BOOST_THRESHOLD 的区域被视为"核心嘴区"，
#         额外叠加 MOUTH_BOOST_STRENGTH 权重，让嘴部动作更突出
#   与全局 gamma 的区别：全局 gamma 同时推高边缘值（可能产生接缝），
#         而 mouth boost 只针对核心嘴区，边缘保持原样
MOUTH_BOOST_STRENGTH = 0.15
MOUTH_BOOST_THRESHOLD = 0.5  # mask 值 > 此阈值才被视为"核心嘴区"


def get_image_blending(image,face,face_box,mask_array,crop_box):
    body = image
    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box
    face_large = copy.deepcopy(body[y_s:y_e, x_s:x_e])
    face_large[y-y_s:y1-y_s, x-x_s:x1-x_s]=face

    mask_image = cv2.cvtColor(mask_array,cv2.COLOR_BGR2GRAY)
    mask_image = (mask_image/255).astype(np.float32)

    # Gamma 校正：增强 mask 对比度，让生成的嘴形更突出
    if MOUTH_MASK_GAMMA != 1.0:
        mask_image = np.power(mask_image, MOUTH_MASK_GAMMA)

    # 嘴部区域定向增强：核心嘴区额外增加权重，边缘保持平滑
    # mask 值 > MOUTH_BOOST_THRESHOLD 的像素被视为核心嘴区
    # mouth_core 从 0（阈值处）到 1（mask=1 处）线性渐变
    if MOUTH_BOOST_STRENGTH > 0:
        mouth_core = np.clip(
            (mask_image - MOUTH_BOOST_THRESHOLD) / (1.0 - MOUTH_BOOST_THRESHOLD + 1e-6),
            0, 1
        )
        mask_image = np.clip(mask_image + MOUTH_BOOST_STRENGTH * mouth_core, 0, 1)

    # mask_not = cv2.bitwise_not(mask_array)
    # prospect_tmp = cv2.bitwise_and(face_large, face_large, mask=mask_array)
    # background_img = body[y_s:y_e, x_s:x_e]
    # background_img = cv2.bitwise_and(background_img, background_img, mask=mask_not)
    # body[y_s:y_e, x_s:x_e] = prospect_tmp + background_img

    #print(mask_image.shape)
    #print(cv2.minMaxLoc(mask_image))

    body[y_s:y_e, x_s:x_e] = cv2.blendLinear(face_large,body[y_s:y_e, x_s:x_e],mask_image,1-mask_image)

    #body.paste(face_large, crop_box[:2], mask_image)
    return body