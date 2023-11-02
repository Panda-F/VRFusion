import os

from scene_understand.monodepth.utils import download_model_if_doesnt_exist


class Config:
    ROOT_PATH = r"F:\Projects\VRFusion\scene_understand\checkpoints"

    instance_split_model = os.path.join(ROOT_PATH, "yolact_im700_54_800000.pth")

    model_name = "mono+stereo_1024x320"

    # download_model_if_doesnt_exist(model_name)
    encoder_path = os.path.join(ROOT_PATH, model_name, "encoder.pth")
    depth_decoder_path = os.path.join(ROOT_PATH, model_name, "depth.pth")
