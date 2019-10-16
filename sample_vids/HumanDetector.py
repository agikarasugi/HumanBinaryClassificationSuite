from __future__ import print_function
import cv2 as cv
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class InvalidBackSubAlgo(Exception):
    def __init__(self):
        super().__init__(self,"Invalid Algorithm for background subtraction")


class HumanDetector:
    def __init__(self, model, algo_type="MOG"):
        self.algo_type = algo_type
        if self.algo_type == "CNT":
            self.backSub = cv.bgsegm.createBackgroundSubtractorCNT() # standard MOG
        elif self.algo_type == "GSOC":
            self.backSub = cv.bgsegm.createBackgroundSubtractorGSOC() # best
        elif self.algo_type == "MOG":
            self.backSub = cv.bgsegm.createBackgroundSubtractorMOG() # fast
        else:
            raise InvalidBackSubAlgo

        self.model = keras.models.load_model(model,
            custom_objects={'leaky_relu': tf.nn.leaky_relu}
        )


    def detect(self, source, img_size=(50, 80), return_image=False):
        """
        Detect human(s) on a video or image stream

        Parameters:
            source: path to video file or device id
            img_size: tuple containing length and width of the desired ROI,
                      make sure the size matches with the pretrained model input
            return_image: boolean value whether to return detected frames or not

        Returns:
            bool: True if human is detected, otherwise False
            np.array: Numpy array of detected frames, only returns if parameter
                      return_image is set to True
        """
        capture = cv.VideoCapture(source)

        human_frames = []
        flag = False

        while True:
            ret, frame = capture.read()
            if frame is None:
                break

            fgMask = self.backSub.apply(frame)

            mask_image = fgMask
            orig_image = frame

            edged = cv.Canny(mask_image, 10, 250)

            _, cnts, _ = cv.findContours(edged.copy(),
                                        cv.RETR_EXTERNAL,
                                        cv.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                x,y,w,h = cv.boundingRect(c)

                if w>25 and h>25:
                    roi_img=orig_image[y:y+h,x:x+w]

                    resized_roi = cv.resize(roi_img,
                                            dsize=(img_size[0], img_size[1]),
                                            interpolation=cv.INTER_CUBIC)

                    resized_roi = resized_roi / 255.0

                    resized_roi = np.array([resized_roi], dtype='float32')
                    classification = self.model.predict(resized_roi)

                    if classification[0][0] > 0.7:
                        human_frames.append(cv.rectangle(
                            orig_image, (x, y), (x+w, y+h), (255,0,0), 2
                        ))
                        flag = True


        if return_image == True:
            human_frames = np.array(human_frames, dtype='float32')/255.0
            return flag, human_frames
        else:
            return flag
