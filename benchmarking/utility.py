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
import matplotlib.pyplot as plt

 

def runner(ip_address, capture_threshold, model, cfg, fps_needed, output_directory,cuda):
    
    current_path = os.path.dirname(os.path.abspath(__file__))
    
    lpr_weights=f'{current_path}/weights/iter2.pth'
    debug_program=False
    
    fields = ['Image_path', 'label', 'time', 'vehicle_type']
    filename=output_directory+'/'+'output.csv'
    with open(filename, 'w') as csvfile:  
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(fields) 
    
    batch_images=[]  
    batch_frame=[]  
    repeat_list_images=[]
    repeat_list_time_stamp=[]
    repeat_list_dictionary=[]
    image_number=0
    
    caught=False
    caught_index=0
    initial_caught_index=0
    i=0;
    cap = cv2.VideoCapture(ip_address)
    incoming_fps=int(cap.get(cv2.CAP_PROP_FPS))
    while(True):
        
        # cap.read(ip_address)
 
        ret, frame = cap.read()
        i=i+1;        
        
        if ret==False:
            break
        
        if(i%(1.0//fps_needed)==0):
            continue
        
        
        
        
        if cuda:
            image = image.cuda()
            
        
        # if(caught==True):
        #     if(i!=caught_index+20):
        #         print(i)
        #         print("chal raha hai")
        #         continue
            
        # caught_index=caught_index+20
        caught_index=i
        frame = cv2.resize(frame, (1000, 400))
        frame1 = frame
        batch_frame.append(frame)
        image = preprocess_image(frame, cfg)
        image=torch.squeeze(image)
        # print(image.shape)
        batch_images.append(image)        
        
                
        if(len(batch_images)>15):   
            with torch.no_grad():
                batch_images_temp=torch.stack(batch_images)
                prediction = model(batch_images_temp, (cfg.dataset.height, cfg.dataset.width))
                prediction = (
                    torch.argmax(prediction["output"][0], dim=1)
                    .cpu()
                    .squeeze(dim=0)
                    .numpy()
                    .astype(np.uint8)
                ) 
                #.reshape(frame.shape[0], frame.shape[1], 1)
                print(prediction.shape)
                print("here")

                for k in range(0,16):
                    # prediction[0].reshape(frame.shape[0], frame.shape[1],1)
                    # prediction = cv2.resize(prediction, (1000, 400))
                    cropped_images, coordinates, centroid = plate_cropper(prediction[k], batch_frame[k])
                    final_image = frame1
                    if len(cropped_images) != 0:
                        labels = get_lprnet_preds(cropped_images, lpr_weights,cuda)
                        repeat_list_dictionary.append(details(prediction[k],labels,coordinates,centroid))
                        repeat_list_images.append(cropped_images)
                        repeat_list_time_stamp.append(datetime.now())
                        if(caught==False):
                            caught=True
                            caught_index=i
                            initial_caught_index=i
                            
                batch_images=[]
                batch_frame=[]
                    
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
            row_contents = [path, label, t, type1+type2]
            append_list_as_row(filename, row_contents)
            image_number=image_number+1
            print('not here')
            caught=False
                
        

