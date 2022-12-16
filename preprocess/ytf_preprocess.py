import os.path
import sys
import argparse
import shutil

parser = argparse.ArgumentParser("Preprocess YTF dataset")
parser.add_argument("--data_dir", type=str)
parser.add_argument("--max_workers", type=int, default=8)

if __name__ == '__main__':

    args = parser.parse_args()

    # copy the metadata (split) to the data_dir
    shutil.copy(os.path.join(os.path.dirname(__file__), "..", "dataset", "misc", "youtube_face", "train_set.csv"),
        args.data_dir)
    shutil.copy(os.path.join(os.path.dirname(__file__), "..", "dataset", "misc", "youtube_face", "val_set.csv"),
        args.data_dir)

    # Crop faces from videos
    sys.path.append(".")
    if not os.path.exists("logs"):
        os.mkdir("logs")

    from util.face_sdk.face_crop import process_images as face_crop_process_images
    face_crop_process_images(
        os.path.join(args.data_dir, "frame_images_DB"),
        os.path.join(args.data_dir, "crop_images_DB"),
        args.max_workers,
    )

    # Face parsing based on these cropped faces
    from util.face_sdk.face_parse import process_images as face_parse_process_images
    face_parse_process_images(
        os.path.join(args.data_dir, "crop_images_DB"),
        os.path.join(args.data_dir, "face_parsing_images_DB")
    )
