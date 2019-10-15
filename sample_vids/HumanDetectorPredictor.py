from __future__ import print_function
import cv2 as cv
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.')
parser.add_argument('--algo', type=str, help='Background subtraction method (MOG, CNT, GSOC).', default='MOG')
args = parser.parse_args()

# create background subtraction object
if args.algo == "MOG":
    backSub = cv.bgsegm.createBackgroundSubtractorMOG() # standard MOG
elif args.algo == "GSOC":
    backSub = cv.bgsegm.createBackgroundSubtractorGSOC() # best
else:
    backSub = cv.bgsegm.createBackgroundSubtractorCNT() # fast


# load the model
model_dir = './models/best.h5'
model = keras.models.load_model(model_dir, custom_objects={'leaky_relu': tf.nn.leaky_relu})

image_size_y = 50
image_size_x = 80

capture = cv.VideoCapture(args.input)

if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    #update the background model
    fgMask = backSub.apply(frame)

    # preview the video
#     cv.imshow('Frame', frame)
#     cv.imshow('FG Mask', fgMask)

    mask_image = fgMask
    orig_image = frame

    edged = cv.Canny(mask_image, 10, 250)
#     cv.imshow("Edges", edged)

    _, cnts, _ = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        cv.drawContours(mask_image, [approx], -1, (0, 255, 0), 2)
        x,y,w,h = cv.boundingRect(c)
        if w>25 and h>25:
            roi_img=orig_image[y:y+h,x:x+w]
#             cv.imshow('ROI', roi_img)

            resized_img = cv.resize(roi_img, dsize=(image_size_y, image_size_x),
                                    interpolation=cv.INTER_CUBIC)
            resized_img = resized_img / 255.0

            resized_img = np.array([resized_img], dtype='float32')
            classification = model.predict(resized_img)

            if classification[0][0] > 0.7:
#                 print(classification[0][0])
#                 cv.imshow('ROI', resized_img[0])
                cv.rectangle(orig_image, (x, y), (x+w, y+h), (255,0,0), 2)
                cv.imshow("Detected", orig_image)

        curr_frame = str(capture.get(cv.CAP_PROP_POS_FRAMES))
        cv.rectangle(orig_image, (10, 2), (100,20), (255,255,255), -1)
        cv.putText(orig_image, curr_frame, (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        cv.imshow("Stream", orig_image)
#         cv.imshow("Mask Stream", mask_image)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
