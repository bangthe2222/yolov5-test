import cv2
import time
import numpy as np
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import time_synchronized
t1=time.time()
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)
device="cpu"
img0= cv2.imread("test.jpg")
img = letterbox(img0, 320, stride=32)[0]
img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
img = np.ascontiguousarray(img)
model = attempt_load("best.pt", map_location="cpu")  # load FP32 model
names = model.module.names if hasattr(model, 'module') else model.names 
def detect(img):

 # get class names
    img = torch.from_numpy(img).to("cpu")
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    t1 = time_synchronized()
    pred = model(img)[0]
    pred = non_max_suppression(pred,float(0.4), float(0.45))
    print(pred)
    t2 = time_synchronized()
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        s =''
        s += '%gx%g ' % img.shape[2:]  # print string
        if len(det):
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum() 
                 # detections per class
                s += f"{n} {names[int(c)]} {'' * (n > 1)}, "  # add to string
        # Print time (inference + NMS)
        print(f'{s}Done. ({t2 - t1:.3f}s)')
detect(img)