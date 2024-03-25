# coding=UTF8
import json
import os
import time

import PIL
import numpy as np
import torch
from torchvision.transforms import transforms

from GlobalConfig import Config
from monodepth import networks
import cv2
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from mmdet.apis import DetInferencer
import pycocotools.mask as mask_utils

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ImageSynthesis:
    def __init__(self, config):
        model = config.instance_model_config
        weights = config.instance_model_weights
        self.segment_inferencer = DetInferencer(model=model, weights=weights)
        self.encoder_path = config.encoder_path
        self.depth_decoder_path = config.depth_decoder_path

    def estimate_instance(self, image):
        seg_res = self.segment_inferencer(image)["predictions"][0]
        labels = seg_res['labels']
        scores = seg_res['scores']
        bboxes = seg_res['bboxes']
        masks = np.array(mask_utils.decode(seg_res['masks']), dtype=np.float32).transpose(2, 0, 1)
        labels, scores, bboxes, masks = self.nms(labels, scores, bboxes, masks, 0.5)
        return labels, scores, bboxes, masks

    def estimate_depth(self, image):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # LOADING PRETRAINED MODEL
        encoder = networks.ResnetEncoder(18, False)
        depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

        loaded_dict_enc = torch.load(self.encoder_path, map_location='cpu')
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)

        loaded_dict = torch.load(self.depth_decoder_path, map_location='cpu')
        depth_decoder.load_state_dict(loaded_dict)

        encoder.eval()
        depth_decoder.eval()

        original_width, original_height = image.size

        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']
        input_image_resized = image.resize((feed_width, feed_height), Image.LANCZOS)

        input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)

        with torch.no_grad():
            features = encoder(input_image_pytorch)
            outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear",
                                                       align_corners=False)

        return disp_resized

    def scene_understand(self, image, is_geo=False):
        # start_time = time.time()
        labels, scores, bboxes, masks = self.estimate_instance(image)
        # end_time = time.time()
        # print("实例分割模块耗时：" + f"{end_time - start_time:.4f}s")
        # start_time = time.time()
        depths = self.estimate_depth(image)[0][0].cuda()
        # end_time = time.time()
        # print("深度估计模块耗时：" + f"{end_time - start_time:.4f}s")
        inst_depths = []
        for mask in masks:
            mask = torch.tensor(mask).cuda()
            pixel_count = mask.sum()
            instances_depth = torch.mul(mask, depths).sum()
            inst_depths.append(instances_depth / pixel_count)
        res = sorted(zip(labels, scores, bboxes, masks, inst_depths), key=lambda x: x[4])
        labels, scores, bboxes, masks, inst_depths = zip(*res)
        return labels, scores, bboxes, masks, inst_depths

    def image_fusion(self, front, background, rate=1.0, x_offset=0, y_offset=500, depth=0.3, score_thresh=0.3):
        labels, scores, bboxes, masks, inst_depths = self.scene_understand(background)
        front = cv2.resize(front, (int(front.shape[1] * rate), int(front.shape[0] * rate)))
        height, width, _ = background.shape
        mask_entire = torch.empty((height, width)).cuda()
        for index, mask in enumerate(masks):
            if scores[index] > score_thresh and inst_depths[index] < depth and labels[index] in [0, 1, 26]:
                mask_entire = torch.logical_or(mask_entire, torch.tensor(mask, device="cuda:0"))
        for y, col in enumerate(front):
            for x, row in enumerate(col):
                # if front[y, x][1] <= 200:
                if sum(front[y, x]) != 0 and (x + x_offset) < width and (y + y_offset) < height and not mask_entire[y + y_offset, x + x_offset]:
                    background[y + y_offset, x + x_offset] = front[y, x]
        return background

    @staticmethod
    def nms(labels, scores, bboxes, masks, iou_thresh):
        """ 非极大值抑制 """
        labels = np.array(labels)
        scores = np.array(scores)
        bboxes = np.array(bboxes)
        masks = np.array(masks)
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        keep = []

        # 按置信度进行排序
        index = np.argsort(scores)[::-1]

        while (index.size):
            # 置信度最高的框
            i = index[0]
            keep.append(index[0])

            if (index.size == 1):  # 如果只剩一个框，直接返回
                break

            # 计算交集左下角与右上角坐标
            inter_x1 = np.maximum(x1[i], x1[index[1:]])
            inter_y1 = np.maximum(y1[i], y1[index[1:]])
            inter_x2 = np.minimum(x2[i], x2[index[1:]])
            inter_y2 = np.minimum(y2[i], y2[index[1:]])
            # 计算交集的面积
            inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
            # 计算当前框与其余框的iou
            iou = inter_area / (areas[index[1:]] + areas[i] - inter_area)
            ids = np.where(iou < iou_thresh)[0]
            index = index[ids + 1]

        return labels[keep], scores[keep], bboxes[keep], masks[keep]

    @staticmethod
    def mask2geojson(mask):
        mask_lst = []
        for row_index, row in enumerate(mask):
            for col_index, col in enumerate(row):
                if col == 1.0:
                    mask_lst.append([col_index, int(0.75 * len(row)) - row_index])
        return mask_lst

    @staticmethod
    def mask_show(image, mask):
        ret, thresh = cv2.threshold(mask.astype(np.uint8), 0.5, 255, 0)
        contours, im = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 第一个参数是轮廓
        res1 = cv2.drawContours(image.copy(), contours=contours, contourIdx=-1, color=(64, 224, 208), thickness=1)
        plt.figure()
        plt.imshow(res1[:, :, ::-1])
        plt.show()


if __name__ == "__main__":
    background = cv2.imread("/home/fangzhou/Project/VRFusion/vrBackend/static/img/frame_back_1.png")
    front = cv2.imread("/home/fangzhou/Project/VRFusion/vrBackend/static/img/frame_front_1.png")

    imageSynthesis = ImageSynthesis(Config)
    # imageSynthesis.scene_understand(background)
    fusion = imageSynthesis.image_fusion(front, background, 0.65, 50, 280, 0.5)
    plt.figure()
    plt.imshow(fusion[:, :, ::-1])
    plt.show()
#     with open('../static/mask/dog_mask.json', "w+", encoding='utf-8') as f:
#         f.write(json.dumps(instances_mask))
