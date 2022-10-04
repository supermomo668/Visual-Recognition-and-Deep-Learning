import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# TODO: given bounding boxes and corresponding scores, perform non max suppression
def nms(bounding_boxes, confidence_score, threshold=0.05):
    """
    bounding boxes of shape     Nx4
    confidence scores of shape  N
    threshold: confidence threshold for boxes to be considered

    return: list of bounding boxes and scores
    """
    iou_thresh = 0.3
    conf_pass_idx = np.where(confidence_score>= threshold)
    bboxes = np.array(bounding_boxes)[conf_pass_idx]
    conf_score = confidence_score[conf_pass_idx]
    order = conf_score.argsort()[::-1]
    # sorted
    bboxes = bboxes[order]
    conf_score = conf_score[confidence_score]
    #
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = conf_score.argsort()[::-1]
    #
    pass_idx = np.arange(len(x1))
    for n, box in enumerate(bboxes):
        # all other idx of boxes to compare 
        compare_idx = compare_idx[compare_idx!=i]
        # Find out the coordinates of the intersection box
        coords = []   # xmin, ymin, xmax, ymax
        for i in range(4):
            coords.append(np.maximum(box[i], boxes[temp_indices,i]))
        # box width and height
        w = np.maximum(0, coords[2] - coords[0] + 1)
        h = np.maximum(0, coords[3] - coords[1] + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[compare_idx]
        # thresholded by iou "high overlaps"
        if np.any(overlap) > iou_thresh:
            pass_idx = pass_idx[pass_idx != n]
    return boxes[pass_idx], conf_score[pass_idx]


# indices = np.arange(len(x1))
#     for i,box in enumerate(boxes):
#         # Create temporary indices  
#         temp_indices = indices[indices!=i]
#         # Find out the coordinates of the intersection box
#         xx1 = np.maximum(box[0], boxes[temp_indices,0])
#         yy1 = np.maximum(box[1], boxes[temp_indices,1])
#         xx2 = np.minimum(box[2], boxes[temp_indices,2])
#         yy2 = np.minimum(box[3], boxes[temp_indices,3])
#         # Find out the width and the height of the intersection box
#         w = np.maximum(0, xx2 - xx1 + 1)
#         h = np.maximum(0, yy2 - yy1 + 1)
#         # compute the ratio of overlap
#         overlap = (w * h) / areas[temp_indices]
#         # if the actual boungding box has an overlap bigger than treshold with any other box, remove it's index  
#         if np.any(overlap) > treshold:
#             indices = indices[indices != i]
#     #return only the boxes at the remaining indices
#     return boxes[indices].astype(int)

# TODO: calculate the intersection over union of two boxes
def iou(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)
    returns IoU vallue
    """
    xmin, ymin, xmax, ymax = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # base case (no overlap)
    if xmax <= xmin2 or xmax2 <= xmin or ymax <= ymin2 or ymax2 <= ymin:
        return 0
    max_intersect_x = np.minimum(xmax, xmax2)
    max_intersect_y = np.minimum(ymax, ymax2)
    min_intersect_x = np.maximum(xmin, xmin2)
    min_intersect_y = np.maximum(ymin, ymin2)
    intersect_A = (max_intersect_x- min_intersect_x)*(max_intersect_y-min_intersect_y)
    #
    max_x, min_x, max_y, min_y =  np.maximum(xmax, xmax2), np.minimum(xmin, xmin2), np.maximum(ymax, ymax2), np.minimum(ymin, ymin2)
    total_A = 2*(max_x-min_x)*(max_y-min_y)
    missing_A = (max_x-max_intersect_x)*(max_y-max_intersect_y)+(min_intersect_x-min_x)*(min_intersect_y-min_y)
    #
    iou = intersect_A/(total_A-intersect_A-missing_A)
    return iou

def batch_iou(boxes1, boxes2):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou

def tensor_to_PIL(image):
    """
    converts a tensor normalized image (imagenet mean & std) into a PIL RGB image
    will not work with batches (if batch size is 1, squeeze before using this)
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255],
    )

    inv_tensor = inv_normalize(image)
    inv_tensor = torch.clamp(inv_tensor, 0, 1)
    original_image = transforms.ToPILImage()(inv_tensor).convert("RGB")

    return original_image


def get_box_data(classes, bbox_coordinates):
    """
    classes : tensor containing class predictions/gt
    bbox_coordinates: tensor containing [[xmin0, ymin0, xmax0, ymax0], [xmin1, ymin1, ...]] (Nx4)

    return list of boxes as expected by the wandb bbox plotter
    """
    box_list = [{
            "position": {
                "minX": bbox_coordinates[i][0],
                "minY": bbox_coordinates[i][1],
                "maxX": bbox_coordinates[i][2],
                "maxY": bbox_coordinates[i][3],
            },
            "class_id": classes[i],
        } for i in range(len(classes))
        ]

    return box_list
