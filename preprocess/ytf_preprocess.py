import os.path
import sys
import argparse

sys.path.append(os.path.join(__file__, "..", "utils", "face_sdk"))
parser = argparse.ArgumentParser("Preprocess YTF dataset")
parser.add_argument("--data_dir", type=str)

if __name__ == '__main__':

    args = parser.parse_args()

    # Crop faces from videos

    try:
        from face_crop import process_images as face_crop_process_images
    except ImportError:
        print("face_crop.py should be placed in utils/face_sdk folder")
        sys.exit(1)

    face_crop_process_images(
        os.path.join(args.data_dir, "frame_images_DB"),
        os.path.join(args.data_dir, "crop_images_DB")
    )

    # Face parsing based on these cropped faces

    try:
        from face_parse import process_images as face_parse_process_images
    except ImportError:
        print("face_parse.py should be placed in utils/face_sdk folder")
        sys.exit(1)

    face_parse_process_images(
        os.path.join(args.data_dir, "crop_images_DB"),
        os.path.join(args.data_dir, "face_parsing_images_DB")
    )
