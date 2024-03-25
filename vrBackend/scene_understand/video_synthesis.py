import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from GlobalConfig import Config
from image_synthesis import ImageSynthesis

# from scene_understand.yolact.infer_instance import InstancePerception

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SceneUnderstand:
    def __init__(self):
        self.imageSynthesis = ImageSynthesis(Config)

    def extract_front(self, input_img):
        resize_img = cv2.resize(input_img, (600, 600))
        rows, cols, channels = resize_img.shape
        roi = resize_img[0:rows, 0:cols]
        img2gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 5, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        input_img_bg = cv2.bitwise_and(roi, roi, mask=mask)
        img2_fg = cv2.bitwise_and(resize_img, resize_img, mask=mask_inv)
        fg_mask = cv2.resize(mask, (input_img.shape[1], input_img.shape[0]))
        # cv2.imshow('mask', fg_mask)
        # cv2.waitKey(0)
        return fg_mask

    def merge_vedio(self, video_bk, video_ft, output_name, x_offset=-100, y_offset=600, rate=1.0, depth=0.0, is_move=False):
        video = cv2.VideoCapture(video_bk)
        video2 = cv2.VideoCapture(video_ft)
        fps = video.get(cv2.CAP_PROP_FPS)  # 帧速率
        fps2 = video2.get(cv2.CAP_PROP_FPS)  # 帧速率

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        output = cv2.VideoWriter(f"../static/output/{output_name}", fourcc, 30, size)
        frame_num = 30 * fps
        frame_num2 = 30 * fps2
        x_offset = x_offset
        y_offset = y_offset
        rate = rate
        depth = depth
        frame_index = 0
        print(f"原视频共{frame_num}帧")
        print(f"待融合视频共{frame_num2}帧")

        while video.isOpened() and frame_num > 0:
            retval, frame_back = video.read()
            try:
                frame_back = cv2.cvtColor(frame_back, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(e)
                frame_num -= 1
                continue
            if retval:
                retval_2, frame_front = video2.read()
                if retval_2 and video2.isOpened():
                    frame_front = cv2.cvtColor(frame_front, cv2.COLOR_RGB2BGR)
                    # 保存单帧图像
                    # cv2.imwrite("/home/fangzhou/Project/VRFusion/vrBackend/static/img/frame_back_1.png", frame_back[:, :, ::-1])
                    # cv2.imwrite("/home/fangzhou/Project/VRFusion/vrBackend/static/img/frame_front_1.png", frame_front[:, :, ::-1])
                    # return
                    merge_frame = self.imageSynthesis.image_fusion(frame_front, frame_back, rate, x_offset, y_offset,
                                                                   depth)
                    output.write(merge_frame[:, :, ::-1])
            frame_num -= 1
            if is_move:
                rate -= 0.001
                x_offset = x_offset + int(pow(1.008, frame_index))
                if frame_index > 180:
                    depth += 0.01
            frame_index += 1
            print(f"剩余帧数{frame_num}")
            # if frame_num == 1200:
            #     break
        video.release()
        video2.release()
        output.release()


if __name__ == "__main__":
    # 蒙古包+野马
    # video_1 = "/home/fangzhou/Project/VRFusion/vrBackend/static/materials/stage_cut.mp4"
    # video_2 = "/home/fangzhou/Project/VRFusion/vrBackend/static/materials/a00c9c1aa46a45fbb4b0250d2c2214d1.mov"
    # output_name = "merge_output.mp4"
    # 音乐节+跳舞
    video_3 = "/home/fangzhou/Project/VRFusion/vrBackend/static/materials/stage_rap_cut.mp4"
    video_4 = "/home/fangzhou/Project/VRFusion/vrBackend/static/materials/dance_1.mp4"
    output_name_1 = "merge_output_1.mp4"
    scene_understand = SceneUnderstand()
    start_time = time.time()
    # scene_understand.merge_vedio(video_1, video_2, output_name, is_move=True)
    scene_understand.merge_vedio(video_3, video_4, output_name_1, 50, 250, 0.65, 0.7)
    print(f"融合共耗时 {time.time() - start_time}")
