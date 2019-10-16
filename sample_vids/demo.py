from HumanDetector import HumanDetector
import cv2

detector = HumanDetector('./models/best.h5', "CNT")

detected, detected_imgs = detector.detect("group_walking.mp4",
                                          return_image=True)

if detected:
    cv2.imshow("Detected {} frames".format(len(detected_imgs)), detected_imgs[0])
    cv2.waitKey(0)
