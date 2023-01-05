"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch

from caspiacraft import CRAFT, craft_utils, imgproc


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def test_net(
    net,
    image_or_file,
    canvas_size=1280,
    mag_ratio=1.5,
    text_threshold=0.7,
    link_threshold=0.4,
    low_text=0.4,
    cuda=True,
    poly=False,
    refine_net=None,
    render_result=False,
):
    """
    Parameters
    ----------
    net
        Torch module responsible for prediction
    image_or_file
        Numpy image or image file on which to run prediction
    canvas_size
        Image size for inference
    mag_ratio
        Image magnification ratio
    text_threshold
        Text confidence threshold
    link_threshold
        Link confidence threshold
    low_text
        Text low-bound score
    cuda
        Use cuda for inference
    poly
        Enable polygon type
    refine_net
        Torch module responsible for link refiner prediction
    render_result
        Render result to image
    """
    if not isinstance(image_or_file, np.ndarray):
        image = imgproc.loadImage(image_or_file)

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = x.unsqueeze(0)  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.to(device="cuda:0")

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()
        poly = True

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    # render results (optional)
    ret_score_heatmap = None
    if render_result:
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_heatmap = imgproc.cvt2HeatmapImg(render_img)

    return polys, ret_score_heatmap


def load_weights_file(model, weights_file, cuda=True):
    device = "cuda:0" if cuda else "cpu"
    model.to(device)
    model.load_state_dict(copyStateDict(torch.load(weights_file, map_location=device)))
    if cuda:
        model = torch.nn.DataParallel(model)
        # cudnn.benchmark = False


def create_net(
    net=None,
    net_weights="weights/craft_mlt_25k.pth",
    cuda=True,
    refiner_weights=None,
):
    # load net
    net = net or CRAFT()  # initialize

    if net_weights:
        print(f"Loading weights from checkpoint ({net_weights})")
        load_weights_file(net, net_weights, cuda)
    net.eval()

    # LinkRefiner
    refine_net = None
    if refiner_weights is not None:
        from refinenet import RefineNet

        refine_net = RefineNet()
        print(f"Loading weights of refiner from checkpoint ({refiner_weights})")
        load_weights_file(refine_net, refiner_weights, cuda)
        refine_net.eval()
        return net, refine_net
    else:
        return net
