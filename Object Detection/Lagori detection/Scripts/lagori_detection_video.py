import cv2
import numpy as np
from matplotlib import pyplot as plt

import cv2
import time
  
  
# define a video capture object
vid = cv2.VideoCapture('D:\Robocon2022\lagori.mp4')

while(True):

    start=time.time() 
    
    ret, img = vid.read()

    def auto_canny(img, sigma=0.33):

        # compute the median of the single channel pixel intensities
            v = np.median(img)
        # apply automatic Canny edge detection using the computed median
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            edged = cv2.Canny(img, lower, upper)
        # return the edged image
            return edged

    # setting threshold of gray image
    # _, threshold = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    image=auto_canny(img,sigma=0.33)
    # using a findContours() function
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # print(contours)
    def find_contour_area(contours):
        area=[]
        for c in contours:
            c_area=cv2.contourArea(c)
            area.append(c_area)
        return area

    print(find_contour_area(contours))


    sorted_area=sorted(contours,key=cv2.contourArea,reverse=True)
 
    i = 0
    for contour in contours:
   
        if i == 0:
            i = 1
            continue
            
            # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            
            # using drawContours() function
        cv2.drawContours(image, [contour], 0, (255, 255, 255), 5)

            # finding center point of shape
        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])

            # putting shape name at center of each shape
        if len(approx) == 3:
            cv2.putText(image, 'Triangle', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif len(approx) == 4:
            cv2.putText(image, 'Quadrilateral', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif len(approx) == 5:
            cv2.putText(image, 'Pentagon', (x, y),	cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif len(approx) == 6:
            cv2.putText(image, 'Hexagon', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        else:
                # area=cv2.contourArea(image)
            cv2.putText(image,"circle" ,(x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
     
    end=time.time()
    # print(end-start)
    cv2.imshow("img",image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()