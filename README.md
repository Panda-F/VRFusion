# VRFusion

## 1.Introduction
基于实例分割和深度估计的二维虚实融合平台

> VRFusion
>> [vrBackend](vrBackend) 后端Django项目
>>> [scene_understand](vrBackend%2Fscene_understand)  场景理解算法
>>>> [GlobalConfig.py](vrBackend%2Fscene_understand%2FGlobalConfig.py)  基础配置文件(权重等)
>>>> [image_synthesis.py](vrBackend%2Fscene_understand%2Fimage_synthesis.py) 单帧融合算法
>>>> [video_synthesis.py](vrBackend%2Fscene_understand%2Fvedio_synthesis.py)  视频融合算法
> 
>> [vrFront](vrFront) 前端Vue项目

## 2.Installation
### 2.1 基础环境
- [ ] python 3.8
- [ ] CUDA 11.3
- [ ] torch 1.10.0
- [ ] torchvision 0.11.0
```
pip install -r requirements.txt
```

## 3.接口说明
### 3.1 [image_synthesis.py](vrBackend%2Fscene_understand%2Fimage_synthesis.py) -> image_fusion
`merge_vedio(self, video_bk, video_ft, output_name, x_offset=-100, y_offset=600, rate=1.0, depth=0.0, is_move=False)`   
#### params:
```
video_bk: 背景视频地址
video_ft: 虚拟素材视频地址
output_name: 输出文件名称 默认保存在vrBackend/static/output文件夹下
x_offset: 虚拟素材水平偏移
y_offset: 虚拟素材垂直偏移
rate=1.0: 虚拟素材缩放比例
depth=0.0: 虚拟素材深度(0-1)
is_move: 虚拟素材是否移动  
```
#### usage:
``` python
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
```

