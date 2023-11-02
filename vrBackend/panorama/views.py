import base64
import json
import os
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from django.http import HttpResponse

from VRFusion import settings
from scene_understand.GlobalConfig import Config
from scene_understand.image_synthesis import ImageSynthesis

# Create your views here.
imageSynthesis = ImageSynthesis(Config)


def image_synthesis(request):
    request_data = json.loads(request.body)
    image = base64.b64decode(request_data['input_image'].split('base64,')[1])
    image = BytesIO(image)
    image = Image.open(image)
    request_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    image_synthesis_obj = ImageSynthesis(Config)
    instances = image_synthesis_obj.occlusion_handling(request_image)
    response_data = {
        "instances": instances,
        "status": "SUCCESS"
    }
    print(json.dumps(response_data, ensure_ascii=False))
    return HttpResponse(json.dumps(response_data, ensure_ascii=False))


def save_img(request):
    file = request.FILES.get('file')
    file_path = os.path.join(settings.BASE_DIR, 'static/media')
    file_name = os.path.join(file_path, file.name)
    with open(file_name, "wb") as f:
        f.write(file.file.read())
    response_data = {
        "code": 200,
        'static_url': "http://127.0.0.1/media/" + file.name,
    }
    return HttpResponse(json.dumps(response_data, ensure_ascii=False))


def fusion_image(request):
    request_data = json.loads(request.body)
    background_path = os.path.join(settings.MEDIA_ROOT, request_data['background'].split('/')[-1])
    front_path = os.path.join(settings.MEDIA_ROOT, request_data['front'].split('/')[-1])
    background = cv2.imread(background_path)
    front = cv2.imread(front_path)
    fusion = imageSynthesis.image_fusion(front, background)
    image_name = "fusionImage.jpg"
    cv2.imwrite(os.path.join(settings.MEDIA_ROOT, image_name), fusion)
    response_data = {
        "code": 200,
        'static_url': "http://127.0.0.1/media/" + image_name,
    }
    return HttpResponse(json.dumps(response_data, ensure_ascii=False))


def test_interface(request):
    with open(r'F:\Projects\VRFusion\static\mask\dog_mask.json', 'r+', encoding="utf-8") as f:
        instances = json.load(f)
        response_data = {
            "instances": instances,
            "status": "SUCCESS"
        }
        print(json.dumps(response_data, ensure_ascii=False))
        return HttpResponse(json.dumps(response_data, ensure_ascii=False))
