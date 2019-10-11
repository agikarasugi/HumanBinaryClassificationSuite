import cv2
import glob

idx = 0
img_files = glob.iglob('*-img.jpg')

for img_file in img_files:
    mask_file = img_file.partition("-")[0]+"-mask.jpg"

    #reading the image 
    image = cv2.imread(mask_file)
    image_src = cv2.imread(img_file)
    edged = cv2.Canny(image, 10, 250)
    # cv2.imshow("Edges", edged)
    # cv2.waitKey(0)
    
    #applying closing function 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("Closed", closed)
    # cv2.waitKey(0)
    
    #finding_contours 
    _, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
        x,y,w,h = cv2.boundingRect(c)
        if w>25 and h>25:
            idx+=1
            new_img=image_src[y:y+h,x:x+w]
            cv2.imwrite('./outs/' + str(idx) + '.png', new_img)
    # cv2.imshow("Output", image)
    # cv2.waitKey(0)