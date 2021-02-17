import numpy as np
import torch
from torch import nn
import argparse
import time
import os
import sys
import cv2
from torchvision import models
from torch import optim
from torch.utils.data import *
import torch.nn.functional as F
from torch.autograd import Variable
from config import get_cfg_defaults
from models.model import create_model
from utils.visualize import visualize
from tqdm import tqdm
from lprnet import *
from utility import *


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="",
        required=True,
        help="Location of current config file",
    )
    parser.add_argument(
        "--path_to_video",
        type=str,
        required=True,
        default="./",
        help="path of video which has to be tested",
    )
    parser.add_argument(
        "--seg_weights",
        type=str,
        required=True,
        default="./",
        help="Path to segmentation weights for which inference needs to be done",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="path to save checkpoints and wandb, final output path will be this path + wandbexperiment name so the output_dir should be root directory",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="",
        required=True,
        help="to read dict with class mapping from data/ folder required so need to create label mapping in beginning",
    )
    parser.add_argument(
        "--lpr_weights", default="./", help="path to trained lprnet weights"
    )
    parser.add_argument("--cuda", default=True, type=bool, help="Use cuda or not")
    parser.add_argument(
        "--debug_program", default=False, type=bool, help="Write video or not"
    )

    args = parser.parse_args()
    return args


def main():
    args = get_parser()
    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(
        [
            "train.config_path",
            args.cfg,
            "train.output_dir",
            args.output_dir,
            "dataset.data_name",
            args.data_name,
        ]
    )
    print(torch.load(os.path.join("data", args.data_name + ".pth")))
    model = create_model(cfg)
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load(args.seg_weights)["state_dict"])
    model.eval()

    print("testing on video")
    directory = r"/media/sanchit/Workspace/Projects/plate_data_download/Videos/combined"
    print(len(os.listdir(directory)))
    for filename in os.listdir(directory):
        current_video = cv2.VideoCapture(os.path.join(directory, filename))
        width = int(current_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(current_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width > 1920:
            width = 1920
            height = 1080
        fps = int(current_video.get(cv2.CAP_PROP_FPS))
        if args.debug_program:
            out_video = cv2.VideoWriter(
                os.path.join(
                    args.output_dir, os.path.join(directory, filename).split("/")[-1]
                ),
                cv2.VideoWriter_fourcc(*"MJPG"),
                fps,
                (
                    width,
                    height,
                ),
            )

        for idx, frame in enumerate(
            tqdm(frame_extract(os.path.join(directory, filename)))
        ):
            if args.debug_program and idx % 2 == 0 and idx != 0:
                out_video.write(final_image)
                continue
            frame = cv2.resize(frame, (1920, 1080))
            frame1 = frame
            image = preprocess_image(frame1, cfg)
            if torch.cuda.is_available():
                image = image.cuda()
            st = time.time()
            prediction = model(image, (frame.shape[0], frame.shape[1]))
            if idx == 0:
                print(time.time() - st)

            prediction = (
                torch.argmax(prediction["output"][0], dim=1)
                .detach()
                .cpu()
                .squeeze(0)
                .numpy()
                .astype(np.uint8)
            ).reshape((frame.shape[0], frame.shape[1], 1))
            # prediction[prediction > 1] = 1
            # newimg=overlay_colour(prediction,frame)
            # out_video.write(newimg, axis=0)
            cropped_images, coordinates, centroid = plate_cropper(prediction, frame)
            final_image = frame1
            if len(cropped_images) != 0:
                labels = get_lprnet_preds(cropped_images, args)
            if args.debug_program:
                if len(cropped_images) != 0:
                    final_image = overlay_colour(prediction, frame, centroid)
                    final_image = write_string(
                        prediction, frame, coordinates, centroid, labels
                    )
                out_video.write(final_image)
        out_video.release()
    return


if __name__ == "__main__":
    main()