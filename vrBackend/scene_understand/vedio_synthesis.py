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

    def merge_vedio(self):
        video = cv2.VideoCapture("../static/materials/Coldplay - Viva La Vida-(1080p).mp4")
        video2 = cv2.VideoCapture("../static/materials/dance.mp4")
        fps = video.get(cv2.CAP_PROP_FPS)  # 帧速率
        fps2 = video2.get(cv2.CAP_PROP_FPS)  # 帧速率

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        output = cv2.VideoWriter("../static/output/merge" + '_output.mp4', fourcc, 30, size)
        frame_num = 30 * fps
        frame_num2 = 30 * fps2
        print(f"原视频共{frame_num}帧")
        print(f"待融合视频共{frame_num2}帧")
        frame_index = 0

        while video.isOpened() and frame_num > 0:
            retval, frame = video.read()
            # try:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if retval:
                retval_2, frame2 = video2.read()
                if retval_2 and video2.isOpened():
                    # 图像理解
                    start_time = time.time()
                    instances_mask = self.imageSynthesis.occlusion_handling(frame, False)
                    mask_entire = torch.tensor(instances_mask[0]["mask"])
                    # 将多个实例相连
                    for index, instance in enumerate(instances_mask[1:]):
                        if instance['score'] >= 0.11:
                            mask = torch.tensor(instance['mask'])
                            mask_entire = torch.logical_or(mask_entire, mask)

                    # 处理前景
                    frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
                    frame2 = cv2.resize(frame2, (int(frame2.shape[1]*0.5), int(frame2.shape[0]*0.5)))
                    # dance_frame_mask = self.extract_front(frame2)
                    x_offset = 200
                    y_offset = 0

                    for x, row in enumerate(frame2):
                        for y, col in enumerate(row):
                            if list(col)[0] < 250 and list(col)[1] < 250 and x + x_offset < frame2.shape[1] and y + y_offset < frame2.shape[0]:
                                if not mask_entire[x + x_offset, y + y_offset]:
                                    frame[x + x_offset, y + y_offset] = frame2[x, y]
                    end_time = time.time()
                    print("图像融合模块耗时：" + f"{end_time - start_time:.4f}s"+f"已处理{30 * fps-frame_num}帧,剩余{frame_num}帧")
                    # plt.figure()
                    # plt.imshow(frame)
                    # plt.show()
                    # return

                    output.write(frame[:, :, ::-1])
            # except Exception as e:
            #     print(e)
            #     continue
            frame_num -= 1
            if frame_num == 800:
                break
        frame_index += 1
        video.release()
        video2.release()
        output.release()


if __name__ == "__main__":
    # img = cv2.imread("../materials/example.png")
    # plt.figure()
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # plt.show()
    scene_understand = SceneUnderstand()
    scene_understand.merge_vedio()
    # video = cv2.VideoCapture("F:\TestProject\VRFusion\materials\dance.mp4")
    # fps = video.get(cv2.CAP_PROP_FPS)
    # print(fps)
