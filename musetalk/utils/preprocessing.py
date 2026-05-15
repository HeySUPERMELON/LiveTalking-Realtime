import sys
from face_detection import FaceAlignment,LandmarksType
from os import listdir, path
import subprocess
import numpy as np
import cv2
import pickle
import os
import json
import platform
import torch
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════
# 面部关键点检测：Linux(Ubuntu) 用 insightface，其他用 DWPose(mmpose)
# ══════════════════════════════════════════════════════════════════

_IS_LINUX = (platform.system() == "Linux")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# --- insightface 模型（仅 Linux 下初始化）---
_insightface_app = None

if _IS_LINUX:
    try:
        import insightface
        from insightface.app import FaceAnalysis
        _insightface_app = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        _insightface_app.prepare(
            ctx_id=0 if device.type == 'cuda' else -1,
            det_size=(640, 640)
        )
        print(f"[preprocessing] 使用 insightface (Linux) 获取面部关键点，设备: {device}")
    except Exception as e:
        print(f"[preprocessing] insightface 初始化失败: {e}，将回退到 DWPose")
        _IS_LINUX = False  # 回退

# --- DWPose 模型（仅非 Linux 下初始化）---
if not _IS_LINUX:
    from mmpose.apis import inference_topdown, init_model
    from mmpose.structures import merge_data_samples
    config_file = './musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
    checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'
    model = init_model(config_file, checkpoint_file, device=device)
    print(f"[preprocessing] 使用 DWPose (mmpose) 获取面部关键点，设备: {device}")


def _get_face_landmarks_68(img_bgr):
    """
    获取 68 个面部关键点 (int32, shape (68, 2))。
    返回 None 表示未检测到人脸。

    Linux  → insightface buffalo_l 的 2d106det (106点) 取前68点
    其他OS → DWPose (mmpose) inference_topdown 取 keypoints[0][23:91]
    """
    if _IS_LINUX:
        # --- insightface 方式 ---
        faces = _insightface_app.get(img_bgr)
        if not faces:
            return None
        # 取最大的人脸（主脸）
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        # landmark_2d_106: shape (106, 2)
        # 前 68 个点的布局与标准 68 点完全一致:
        #   jaw(0-16), leb(17-21), reb(22-26), nose(27-35),
        #   leye(36-41), reye(42-47), olip(48-59), ilip(60-67)
        pts68 = face.landmark_2d_106[:68].astype(np.int32)
        return pts68
    else:
        # --- DWPose 方式 ---
        results = inference_topdown(model, img_bgr)
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark = keypoints[0][23:91].astype(np.int32)
        return face_land_mark


# initialize the face detection model
fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=str(device))

# maker if the bbox is not sufficient
coord_placeholder = (0.0,0.0,0.0,0.0)

def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def get_bbox_range(img_list,upperbondrange =0):
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:',upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    for fb in tqdm(batches):
        img = np.asarray(fb)[0]

        # 统一接口获取 68 点
        face_land_mark = _get_face_landmarks_68(img)
        if face_land_mark is None:
            coords_list += [coord_placeholder]
            continue

        # get bounding boxes by face detection
        bbox = fa.get_detections_for_batch(np.asarray(fb))

        # adjust the bounding box refer to landmark
        for j, f in enumerate(bbox):
            if f is None: # no face in the image
                coords_list += [coord_placeholder]
                continue

            half_face_coord = face_land_mark[29]
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange + half_face_coord[1]

    text_range=f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}"
    return text_range


def get_landmark_and_bbox(img_list,upperbondrange =0):
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:',upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    for fb in tqdm(batches):
        img = np.asarray(fb)[0]

        # 统一接口获取 68 点
        face_land_mark = _get_face_landmarks_68(img)
        if face_land_mark is None:
            coords_list += [coord_placeholder]
            continue

        # get bounding boxes by face detection
        bbox = fa.get_detections_for_batch(np.asarray(fb))

        # adjust the bounding box refer to landmark
        for j, f in enumerate(bbox):
            if f is None: # no face in the image
                coords_list += [coord_placeholder]
                continue

            half_face_coord = face_land_mark[29]
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange + half_face_coord[1]
            half_face_dist = np.max(face_land_mark[:,1]) - half_face_coord[1]
            min_upper_bond = 0
            upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)

            f_landmark = (np.min(face_land_mark[:, 0]),int(upper_bond),np.max(face_land_mark[:, 0]),np.max(face_land_mark[:,1]))
            x1, y1, x2, y2 = f_landmark

            if y2-y1<=0 or x2-x1<=0 or x1<0: # if the landmark bbox is not suitable, reuse the bbox
                coords_list += [f]
                w,h = f[2]-f[0], f[3]-f[1]
                print("error bbox:",f)
            else:
                coords_list += [f_landmark]

    print("********************************************bbox_shift parameter adjustment**********************************************************")
    print(f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}")
    print("*************************************************************************************************************************************")
    return coords_list,frames


if __name__ == "__main__":
    img_list = ["./results/lyria/00000.png","./results/lyria/00001.png","./results/lyria/00002.png","./results/lyria/00003.png"]
    crop_coord_path = "./coord_face.pkl"
    coords_list,full_frames = get_landmark_and_bbox(img_list)
    with open(crop_coord_path, 'wb') as f:
        pickle.dump(coords_list, f)

    for bbox, frame in zip(coords_list,full_frames):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        print('Cropped shape', crop_frame.shape)

        #cv2.imwrite(path.join(save_dir, '{}.png'.format(i)),full_frames[i][0][y1:y2, x1:x2])
    print(coords_list)
