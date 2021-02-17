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
from dataset.augmentations import normalize
from torch.autograd import Variable
from config import get_cfg_defaults
from models.model import create_model
from utils.visualize import visualize
from tqdm import tqdm
from matplotlib import cm
from lprnet import *
from copy import deepcopy




def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",type=str,default="",required=True,help="Location of current config file")
    parser.add_argument("--path_to_video",type=str,required=True,default="./",help="path of video which has to be tested")
    parser.add_argument("--seg_weights",type=str,required=True,default="./",help="Path to segmentation weights for which inference needs to be done")
    parser.add_argument("--output_dir",type=str,default="./",help="path to save checkpoints and wandb, final output path will be this path + wandbexperiment name so the output_dir should be root directory")
    parser.add_argument("--data_name",type=str,default="",required=True,help="to read dict with class mapping from data/ folder required so need to create label mapping in beginning")
    parser.add_argument('--lpr_weights', default='./', help='path to trained lprnet weights')
    parser.add_argument('--cuda', default=False, type=bool, help='Use cuda or not')
    parser.add_argument('--debug_program',default=True, type=bool,help='Write video or not')

    args = parser.parse_args()
    return args

def frame_extract(path):
    vidObj = cv2.VideoCapture(path)
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            yield image
            
def plate_cropper(image,imagergb):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    size_factor=1.2
    cropped_images = []
    top_left_coordinates=[]
    for c in cnts:
        area=cv2.contourArea(c)
        if area<400:
            continue
        
        temp_rect=[]
        rect = cv2.minAreaRect(c)
        temp_rect.append(rect[0][0])
        temp_rect.append(rect[0][1])
        temp_rect.append(rect[1][0]*size_factor)
        temp_rect.append(rect[1][1]*size_factor)
        temp_rect.append(rect[2])
        rect=((temp_rect[0],temp_rect[1]),(temp_rect[2],temp_rect[3]),temp_rect[4])
        top_left_coordinates.append([rect[0][0]-rect[1][0]//2, rect[0][1]-rect[1][1]//2])
        
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(imagergb, [box], 0, (0, 255, 0), 3)

        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been straightened
        #we have to filter boxes based on size, if box size is too small then discard them.
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        warped = cv2.warpPerspective(imagergb, M, (width, height))
        if width<height:
            warped=cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cropped_images.append(warped)
        # cv2.imwrite('/home/sanchit/Desktop/warped'+'/'+str(i)+str(width)+'.jpg',warped)
        # i=i+1
    
    return cropped_images, top_left_coordinates

def overlay_colour(prediction, frame):
    N = 11 
    colours = cm.get_cmap('viridis', N)  
    cmap = colours(np.linspace(0, 1, N))  # Obtain RGB colour map
    cmap[0,-1] = 0  # Set alpha for label 0 to be 0
    cmap[1:,-1] = 0.3  # Set the other alphas for the labels to be 0.3
    output = cmap[prediction.flatten()]
    R, C = prediction.shape[:2]
    output = output.reshape((R, C, -1))
    alpha = output[:,:,3]
    A1 = output[:,:,:3]
    im = np.multiply(A1, alpha.reshape(R,C,1)) + np.multiply(frame, 1-alpha.reshape(R,C,1))
    return im

def write_string(image,top_left_coordinates,labels):
    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 0.5
    color = (0, 0, 0)
    thickness = 1
    for i in range(len(top_left_coordinates)):
        
        org = (int(top_left_coordinates[i][0]),int(top_left_coordinates[i][1])) 
        image[org[1]-20:org[1],org[0]:org[0]+100,:] = (255,255,255) 
        image = cv2.putText(image, labels[i], org, font, fontScale, color, thickness, cv2.LINE_AA) 

    return image

def preprocess_image(image, cfg):
    image = normalize(image, cfg)
    return torch.unsqueeze(image, dim=0)

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
    model = nn.DataParallel(model)
    print(torch.load(args.seg_weights).keys())
    model.load_state_dict(torch.load(args.seg_weights)["state_dict"])
    model.eval()
    
    print("testing on video")
    # current_video = cv2.VideoCapture(args.path_to_video)
    directory=''
    for filename in os.listdir(directory):
        current_video= cv2.VideoCapture(os.path.join(directory,filename))
        width = int(current_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(current_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if(width>1920):
            width=1920
            height=1080
        fps = int(current_video.get(cv2.CAP_PROP_FPS))
        if args.debug_program:
            out_video = cv2.VideoWriter(
                os.path.join('/home/sanchit/Desktop/semantic-segmentation-pipeline-master/output videos', os.path.join(directory,filename).split("/")[-1]),
                cv2.VideoWriter_fourcc(*"MJPG"),
                fps,
                (
                    width,
                    height,
                ), 
            )
        for idx, frame in enumerate(tqdm(frame_extract(os.path.join(directory,filename)))):
            frame = cv2.resize(frame,(1920,1080))
            frame1 = frame
            image = preprocess_image(frame1, cfg)
            if torch.cuda.is_available():
                image = image.cuda()
            st = time.time()
            prediction = model(image, (frame.shape[0], frame.shape[1]))
            if idx == 0:
                print(time.time()-st)

            prediction = (
                torch.argmax(prediction["output"][0], dim=1)
                .detach()
                .cpu()
                .squeeze(0)
                .numpy()
                .astype(np.uint8)
            ).reshape((frame.shape[0], frame.shape[1],1))
            prediction[prediction>1] = 1
            # import matplotlib.pyplot as plt
            # plt.imshow(prediction*255)
            # plt.show()

            # newimg=overlay_colour(prediction,frame)
            # out_video.write(newimg, axis=0)
            cropped_images, top_left_coordinates = plate_cropper(frame*prediction, frame)
            
            # import matplotlib.pyplot as plt
            # for i in cropped_images:
            #     plt.imshow(i)
            #     plt.show()
            final_image=frame1
            if len(cropped_images) != 0:
                labels = get_lprnet_preds(cropped_images, args)
            if args.debug_program:
                if len(cropped_images) != 0:
                    final_image=write_string(frame,top_left_coordinates,labels)
                # cv2.imshow('video',final_image)
                # if cv2.waitKey(10) == 27:
                #     break
                out_video.write(final_image)
        out_video.release()
    return

if __name__ == "__main__":
    main()