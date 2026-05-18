import numpy as np
import cv2
import copy

# ──────────────────────────────────────────────────────────────────────
# Mask gamma 校正系数 — 控制嘴形可见幅度
#   1.0 = 原始（无修改），0.5 = 大幅增强嘴形，0.3 = 极度增强（可能有接缝）
#   原理：mask 边缘值 0.5 → gamma(0.5,0.5)=0.71 → 生成的嘴占比从 50% 提升到 71%
#   对于"嘴形幅度不够"的问题，降低此值可让嘴部动作更明显
# ──────────────────────────────────────────────────────────────────────
MOUTH_MASK_GAMMA = 0.6


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