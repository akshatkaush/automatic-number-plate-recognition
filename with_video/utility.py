import cv2
import numpy as np
from matplotlib import cm
import argparse
import os
import torch
import copy
from dataset.augmentations import normalize

classes = {
    10: [(102, 102, 0), "auto front"],
    9: [(104, 104, 104), "auto back"],
    8: [(255, 102, 102), "bus front"],
    7: [(255, 255, 0), "bus back"],
    5: [(255, 0, 127), "truck back"],
    6: [(204, 0, 204), "truck front"],
    4: [(102, 204, 0), "bike front"],
    2: [(0, 0, 255), "car front"],
    3: [(0, 255, 0), "bike back"],
    1: [(255, 0, 0), "car back"],
    0: [(0, 0, 0), "background"],
}


def frame_extract(path):
    vidObj = cv2.VideoCapture(path)
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            yield image


def plate_cropper(image, imagergb):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    size_factor = 1.03
    cropped_images = []
    coordinates = []
    centroid = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 400:
            continue

        temp_rect = []
        rect = cv2.minAreaRect(c)
        centroid.append(rect)
        temp_rect.append(rect[0][0])
        temp_rect.append(rect[0][1])
        temp_rect.append(rect[1][0] * size_factor)
        temp_rect.append(rect[1][1] * size_factor)
        temp_rect.append(rect[2])
        rect = (
            (temp_rect[0], temp_rect[1]),
            (temp_rect[2], temp_rect[3]),
            temp_rect[4],
        )

        box = cv2.boxPoints(rect)
        box = np.int0(box)
        coordinates.append(box)
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        dst_pts = np.array(
            [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped = cv2.warpPerspective(imagergb, M, (width, height))
        if width < height:
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cropped_images.append(warped)
        # cv2.imwrite('/home/sanchit/Desktop/warped'+'/'+str(i)+str(width)+'.jpg',warped)
        # i=i+1

    return cropped_images, coordinates, centroid


def overlay_colour(prediction, frame, centroid):

    temp_img = frame.copy()
    for i in range(len(centroid)):
        temp = centroid[i]
        pred_class = prediction[int(temp[0][1])][int(temp[0][0])][0]
        box = cv2.boxPoints(temp)
        box = np.int0(box)
        cv2.drawContours(temp_img, [box], 0, classes[pred_class][0], -1)
        # frame = cv2.putText(image, 'OpenCV', org, font,
        #            fontScale, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(temp_img, 0.5, frame, 0.5, 0, frame)
    return frame


def write_string(prediction, image, coordinates, centroid, labels):
    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 0.5
    color = (0, 0, 0)
    thickness = 1

    for i in range(len(coordinates)):
        temp = centroid[i]
        # box = cv2.boxPoints(temp)
        # box = np.int0(box)
        pred_class = prediction[int(temp[0][1])][int(temp[0][0])][0]
        cv2.drawContours(image, [coordinates[i]], 0, (0, 255, 0), 3)

        org = (
            int(temp[0][0] - temp[1][0] // 2),
            int(temp[0][1] - temp[1][1] // 2),
        )
        del temp
        image[org[1] - 25 : org[1], org[0] : org[0] + 200, :] = (255, 255, 255)
        image = cv2.putText(
            image,
            labels[i] + "  " + classes[pred_class][1],
            org,
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    # cv2.addWeighted(image_temp, 0.4, image, 0.6, 0, image)

    return image


def preprocess_image(image, cfg):
    image = normalize(image, cfg)
    return torch.unsqueeze(image, dim=0)
