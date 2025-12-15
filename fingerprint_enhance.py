import cv2
import numpy as np
from rembg import remove
from PIL import Image
import os

def enhance_fingerprint(image_path, out_path):
    img = Image.open(image_path)
    img = remove(img)
    img = np.array(img)

    alpha = img[:,:,3]
    rgb = img[:,:,:3]
    white = np.ones_like(rgb)*255
    img = (rgb*(alpha[:,:,None]/255) + white*(1-alpha[:,:,None]/255)).astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)

    clahe = cv2.createCLAHE(2.0, (8,8))
    enh = clahe.apply(blur)

    binary = cv2.adaptiveThreshold(
        enh, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 2
    )

    cv2.imwrite(out_path, binary)
