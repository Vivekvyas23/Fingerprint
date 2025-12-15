import cv2
import numpy as np
import tensorflow as tf
import os

IMG_SIZE = 640
CONF_THRES = 0.5
IOU_THRES = 0.4

def xywh2xyxy(b):
    cx, cy, w, h = b
    return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]

def iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:,0])
    y1 = np.maximum(box[1], boxes[:,1])
    x2 = np.minimum(box[2], boxes[:,2])
    y2 = np.minimum(box[3], boxes[:,3])
    inter = np.maximum(0, x2-x1)*np.maximum(0, y2-y1)
    a1 = (box[2]-box[0])*(box[3]-box[1])
    a2 = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    return inter / (a1 + a2 - inter + 1e-8)

def nms(boxes, scores, th):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs):
        i = idxs[0]
        keep.append(i)
        idxs = idxs[1:][iou(boxes[i], boxes[idxs[1:]]) < th]
    return keep

def detect_and_crop(image_path, model_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    img0 = cv2.imread(image_path)
    H, W = img0.shape[:2]

    img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    img = np.expand_dims(img, 0)

    interpreter.set_tensor(inp["index"], img)
    interpreter.invoke()
    raw = interpreter.get_tensor(out["index"])

    if raw.ndim == 2:
        arr = raw
    else:
        axis = [i for i,s in enumerate(raw.shape) if s == 5][0]
        arr = np.moveaxis(raw, axis, -1).reshape(-1, 5)

    boxes, scores = arr[:, :4], arr[:, 4]
    mask = scores > CONF_THRES
    boxes, scores = boxes[mask], scores[mask]

    boxes = np.array([xywh2xyxy(b) for b in boxes])
    boxes[:, [0,2]] *= W
    boxes[:, [1,3]] *= H
    boxes = boxes.clip([0,0,0,0], [W-1,H-1,W-1,H-1])

    keep = nms(boxes, scores, IOU_THRES)

    crops = []
    for i, k in enumerate(keep):
        x1,y1,x2,y2 = boxes[k].astype(int)
        crop = img0[y1:y2, x1:x2]
        path = f"{out_dir}/finger_{i+1}.jpg"
        cv2.imwrite(path, crop)
        crops.append(path)

    return crops
