import glob
import logging.config
import os.path
import sys
from multiprocessing import set_start_method
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from util.face_sdk.core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from util.face_sdk.core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from util.face_sdk.core.model_handler.face_parsing.FaceParsingModelHandler import FaceParsingModelHandler
from util.face_sdk.core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from util.face_sdk.core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from util.face_sdk.core.model_loader.face_parsing.FaceParsingModelLoader import FaceParsingModelLoader

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

logging.config.fileConfig(os.path.join("util", "face_sdk", "config", "logging.conf"))
logger = logging.getLogger('api')

with open(os.path.join("util", "face_sdk", "config", "model_conf.yaml")) as f:
    model_conf = yaml.load(f, Loader=yaml.FullLoader)

# common setting for all models, need not modify.
model_path = os.path.join("util", "face_sdk", "models")
sys.path.append(os.path.join("util", "face_sdk"))
# face detection model setting.
scene = 'non-mask'
model_category = 'face_detection'
model_name = model_conf[scene][model_category]
logger.info('Start to load the face detection model...')
faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
model, cfg = faceDetModelLoader.load_model()
faceDetModelHandler = FaceDetModelHandler(model, 'cuda:0', cfg)

# face landmark model setting.
model_category = 'face_alignment'
model_name = model_conf[scene][model_category]
logger.info('Start to load the face landmark model...')
faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
model, cfg = faceAlignModelLoader.load_model()
faceAlignModelHandler = FaceAlignModelHandler(model, 'cuda:0', cfg)

# face parsing model setting.
scene = 'non-mask'
model_category = 'face_parsing'
model_name = model_conf[scene][model_category]
logger.info('Start to load the face parsing model...')
faceParsingModelLoader = FaceParsingModelLoader(model_path, model_category, model_name)
model, cfg = faceParsingModelLoader.load_model()
faceParsingModelHandler = FaceParsingModelHandler(model, 'cuda:0', cfg)


def parse_face_img(img_path: str, output_path: str):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    dets = faceDetModelHandler.inference_on_image(image)
    face_nums = dets.shape[0]
    with torch.no_grad():
        for i in range(face_nums):
            landmarks = faceAlignModelHandler.inference_on_image(image, dets[i])

            landmarks = torch.from_numpy(landmarks[[104, 105, 54, 84, 90]]).float()
            if i == 0:
                landmarks_five = landmarks.unsqueeze(0)
            else:
                landmarks_five = torch.cat([landmarks_five, landmarks.unsqueeze(0)], dim=0)
        try:
            faces = faceParsingModelHandler.inference_on_image(face_nums, image, landmarks_five)["seg"]["logits"].cpu()
            faces = faces.softmax(dim=1).argmax(dim=1).numpy()
        except UnboundLocalError:
            faces = np.zeros((0, 224, 224), dtype="int64")
        np.save(output_path, faces)


def check_exists(output_path: str):
    return os.path.exists(output_path)


def process_images(image_path: str, output_path: str):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    files = glob.glob(f"{image_path}/*/*/*.jpg")

    for i, file in enumerate(tqdm(files)):
        save_path = file.replace(image_path, output_path).replace(".jpg", ".npy")
        Path("/".join(save_path.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
        parse_face_img(file, save_path)
