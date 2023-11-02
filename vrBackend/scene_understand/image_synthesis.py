# coding=UTF8
import json
import os
import time

import PIL
import numpy as np
import torch
from torchvision.transforms import transforms

from scene_understand.GlobalConfig import Config
from scene_understand.monodepth import networks
from scene_understand.yolact.data import cfg
import cv2
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

from scene_understand.yolact.infer_instance import InstancePerception

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ImageSynthesis:
    def __init__(self, config):
        self.instance_model = config.instance_split_model
        self.encoder_path = config.encoder_path
        self.depth_decoder_path = config.depth_decoder_path

    def estimate_instance(self, image):
        instance_perception = InstancePerception(self.instance_model)
        classes, scores, boxes, masks = instance_perception.infer(image)
        instance_lst = []
        for index, mask in enumerate(masks):
            instance_dict = {'mask': mask.cuda(), 'score': scores[index],
                             'class': cfg.dataset.class_names[classes[index]], 'box': masks[index]}
            instance_lst.append(instance_dict)
        return instance_lst

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

    def occlusion_handling(self, image, is_geo=False):
        start_time = time.time()
        instances = self.estimate_instance(image)
        end_time = time.time()
        print("实例分割模块耗时：" + f"{end_time - start_time:.4f}s")
        start_time = time.time()
        depth = self.estimate_depth(image)[0][0].cuda()
        end_time = time.time()
        print("深度估计模块耗时：" + f"{end_time - start_time:.4f}s")
        for i in instances:
            mask = i['mask'].cuda()
            pixel_count = mask.sum()
            instances_depth = torch.mul(mask, depth).sum()
            i['depth'] = instances_depth / pixel_count
            i['depth'] = float(i['depth'].cpu().numpy())
            i['score'] = float(i['score'].cpu().numpy())
        instances = sorted(instances, key=lambda x: x['depth'])
        if not is_geo:
            return instances
        else:
            for instance in instances:
                instance['mask'] = self.mask2geojson(instance['mask'])
            return instances

    def image_fusion(self, front, background, rate=0.5, x_offset=250, y_offset=150, depth=-1):
        instances_mask = self.occlusion_handling(background, False)
        front = cv2.resize(front, (int(front.shape[1] * rate), int(front.shape[0] * rate)))
        height, width, _ = background.shape
        mask_entire = torch.empty((height, width)).cuda()
        for index, instance in enumerate(instances_mask):
            if instance['score'] >= 0.11:
                mask = torch.tensor(instance['mask'])
                mask_entire = torch.logical_or(mask_entire, mask)
        for x, col in enumerate(front):
            for y, row in enumerate(col):
                if (list(row)[0] > 0 or list(row)[1] > 0 or list(row)[2] > 0) and x + x_offset < height and y + y_offset < width:
                    if not mask_entire[x + y_offset, y + x_offset]:
                        background[x + y_offset, y + x_offset] = front[x, y]
        return background

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
    background = cv2.imread("C:/Users/24422/Desktop/grassland.jpg")
    front = cv2.imread("C:/Users/24422/Desktop/people.png")

    imageSynthesis = ImageSynthesis(Config)
    fusion = imageSynthesis.image_fusion(front, background)
    plt.figure()
    plt.imshow(fusion[:, :, ::-1])
    plt.show()
#     with open('../static/mask/dog_mask.json', "w+", encoding='utf-8') as f:
#         f.write(json.dumps(instances_mask))
