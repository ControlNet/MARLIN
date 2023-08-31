import glob
import logging.config
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Tuple

import cv2
import ffmpeg
import yaml
from numpy import ndarray
from tqdm.auto import tqdm

from marlin_pytorch.util import crop_with_padding
from util.face_sdk.core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from util.face_sdk.core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader

logging.config.fileConfig(os.path.join("util", "face_sdk", "config", "logging.conf"))
logger = logging.getLogger('api')

with open(os.path.join("util", "face_sdk", "config", "model_conf.yaml")) as f:
    model_conf = yaml.load(f, Loader=yaml.FullLoader)

# common setting for all model, need not modify.
model_path = os.path.join("util", "face_sdk", 'models')

# model setting, modified along with model
scene = 'non-mask'
model_category = 'face_detection'
model_name = model_conf[scene][model_category]

logger.info('Start to load the face detection model...')
# load model
sys.path.append(os.path.join("util", "face_sdk"))
faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
model, cfg = faceDetModelLoader.load_model()
faceDetModelHandler = FaceDetModelHandler(model, 'cuda:0', cfg)


def crop_face(frame, margin=1, x=0, y=0) -> Tuple[ndarray, int, int, int]:
    assert frame.ndim == 3 and frame.shape[2] == 3, "frame should be 3-dim"
    dets = faceDetModelHandler.inference_on_image(frame)
    if len(dets) > 0:
        x1, y1, x2, y2, confidence = dets[0]
        # center
        x, y = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        margin = int(max(abs(x2 - x1), abs(y2 - y1)) / 2)
    # crop face
    face = crop_with_padding(frame, x - margin, x + margin, y - margin, y + margin, 0)
    face = cv2.resize(face, (224, 224))
    return face, margin, x, y


def crop_face_video(video_path: str, save_path: str, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=30) -> None:
    cap = cv2.VideoCapture(video_path)
    writer = cv2.VideoWriter(save_path, fourcc=fourcc, fps=fps, frameSize=(224, 224))
    x, y = 0, 0
    margin = 1

    while True:
        ret, frame = cap.read()
        if ret:
            face, margin, x, y = crop_face(frame, margin, x, y)
            writer.write(face)
        else:
            break

    cap.release()
    writer.release()


def crop_face_img(img_path: str, save_path: str):
    frame = cv2.imread(img_path)
    face = crop_face(frame)[0]
    cv2.imwrite(save_path, face)


def process_videos(video_path, output_path, ext="mp4", max_workers=8):
    if ext == "mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif ext == "avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        raise ValueError("ext should be mp4 or avi")

    Path(output_path).mkdir(parents=True, exist_ok=True)

    files = os.listdir(video_path)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for f_name in tqdm(files):
            if f_name.endswith('.' + ext):
                source_path = os.path.join(video_path, f_name)
                target_path = os.path.join(output_path, f_name)
                fps = eval(ffmpeg.probe(source_path)["streams"][0]["avg_frame_rate"])
                futures.append(executor.submit(crop_face_video, source_path, target_path, fourcc,
                    fps))

        for future in tqdm(futures):
            future.result()


def process_images(image_path: str, output_path: str, max_workers: int = 8):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    files = glob.glob(f"{image_path}/*/*/*.jpg")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for file in tqdm(files):
            save_path = file.replace(image_path, output_path)
            Path("/".join(save_path.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
            futures.append(executor.submit(crop_face_img, file, save_path))

        for future in tqdm(futures):
            future.result()
