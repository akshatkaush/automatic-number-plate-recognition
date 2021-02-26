import torch
import numpy as np
from utility import runner
from PIL import Image
import os
from preload import preloader
import cv2

def process(ip_address='C:/Users/Akshat/Downloads/VID_20201027_115407.mp4', output_directory='./', fps_needed=0.5, capture_threshold=4):
    current_path = os.path.dirname(os.path.abspath(__file__))
    is_cuda=False
    model,cfg=preloader(is_cuda)
    with torch.no_grad():
        runner(ip_address, capture_threshold, model, cfg, fps_needed,output_directory,is_cuda)
            
if __name__ == "__main__":
    process()