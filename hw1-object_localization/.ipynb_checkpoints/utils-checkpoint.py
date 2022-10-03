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
    bounding_boxes = np.array(bounding_boxes)  # array
    order = confidence_score.argsort()[::-1]
    # sorted
    bounding_boxes = bounding_boxes[order]
    confidence_score = confidence_score[confidence_score]
    #
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = confidence_score.argsort()[::-1]
    #
    filtered_bboxes = []
    while len(order) > 0:
        i = order[0]
        filtered_bboxes.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
    boxes, scores = None, None
    return boxes, scores


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
