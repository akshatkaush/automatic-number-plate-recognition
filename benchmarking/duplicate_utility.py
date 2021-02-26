import numpy as np
import torch
import time
import os
import sys
import cv2
from lprnet import *
from dataset.augmentations import *
from func import *
from datetime import datetime
import csv

 

def runner(ip_address, capture_threshold, model, cfg, fps_needed, output_directory,cuda):
    
    current_path = os.path.dirname(os.path.abspath(__file__))
    
    lpr_weights=f'{current_path}/weights/iter2.pth'
    debug_program=False
    
    fields = ['Image_path', 'label', 'time', 'vehicle_type']
    filename=output_directory+'/'+'output.csv'
    with open(filename, 'w') as csvfile:  
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(fields) 
         
    repeat_list_images=[]
    repeat_list_time_stamp=[]
    repeat_list_dictionary=[]
    image_number=0
    
    caught=False
    caught_index=0
    initial_caught_index=0
    i=0;
    
    while(True):
        cap = cv2.VideoCapture(ip_address)
        # cap.read(ip_address)
        incoming_fps=int(cap.get(cv2.CAP_PROP_FPS)) 
        ret, frame = cap.read()
        i=i+1;        
        
        if ret==False:
            break
        
        if(i%(1.0//fps_needed)==0):
            continue
        
        frame = cv2.resize(frame, (1920, 1080))
        frame1 = frame
        image = preprocess_image(frame, cfg)
        # if torch.cuda.is_available():
        #     image = image.cuda()
        if(caught==True):
            if(i!=caught_index+20):
                print(i)
                continue
            else:
                caught_index=caught_index+20
            
        
        with torch.no_grad():
            prediction = model(image, (cfg.dataset.height, cfg.dataset.width))
            prediction = (
                torch.argmax(prediction["output"][0], dim=1)
                .cpu()
                .squeeze(dim=0)
                .numpy()
                .astype(np.uint8)
            ).reshape(frame.shape[0], frame.shape[1])

            cropped_images, coordinates, centroid = plate_cropper(prediction, frame)
            final_image = frame1
            if len(cropped_images) != 0:
                labels = get_lprnet_preds(cropped_images, lpr_weights,cuda)
                repeat_list_dictionary.append(details(prediction,labels,coordinates,centroid))
                repeat_list_images.append(cropped_images)
                repeat_list_time_stamp.append(datetime.now())
                if(caught==False):
                    caught=True
                    caught_index=i
                    initial_caught_index=i
            
            if(caught==True and (caught_index-initial_caught_index)>=incoming_fps*capture_threshold):
                im=repeat_list_images[len(repeat_list_images)//2][0]
                t=repeat_list_time_stamp[len(repeat_list_time_stamp)//2]
                label=(repeat_list_dictionary[len(repeat_list_dictionary)//2])[0][3]
                type1=(repeat_list_dictionary[len(repeat_list_dictionary)//2])[0][1]
                type2=(repeat_list_dictionary[len(repeat_list_dictionary)//2])[0][2]
                repeat_list_images=[]
                repeat_list_time_stamp=[]
                repeat_list_dictionary=[]
                path=output_directory+'/'+str(image_number)+'.jpg'
                cv2.imwrite(path,im)
                print()
                row_contents = [path, label, t, type1+type2]
                append_list_as_row(filename, row_contents)
                image_number=image_number+1
                print('not here')
                caught=False