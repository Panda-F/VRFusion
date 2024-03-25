import os

from monodepth.utils import download_model_if_doesnt_exist


class Config:
    ROOT_PATH = r"/home/fangzhou/Checkpoints"

    instance_split_model = os.path.join(ROOT_PATH, "yolact_im700_54_800000.pth")

    model_name = "mono+stereo_1024x320"

    # download_model_if_doesnt_exist(model_name)
    encoder_path = os.path.join(ROOT_PATH, model_name, "encoder.pth")
    depth_decoder_path = os.path.join(ROOT_PATH, model_name, "depth.pth")

    coco_label = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
                  "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
                  "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                  "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
                  "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                  "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
                  "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
