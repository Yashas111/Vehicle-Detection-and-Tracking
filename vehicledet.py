import numpy as np
import cv2

#capture the video
cap = cv2.VideoCapture("indian traffic1.mp4")

#get the number of frames,fps,etc.
frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

width = int(width)
    
height = int(height)
    
print(frames_count, fps, width, height)

sub = cv2.createBackgroundSubtractorMOG2()  # create background

#inform to start saving a video file
ret, frame = cap.read()  # import image
ratio = 1.0  

#get the width,height and channels of the frames
width2, height2, channels = frame.shape

#save the video in anew file
video = cv2.VideoWriter('traffic_counter.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (height2, width2), 1)

while True:
    ret, frame = cap.read()  # import image and check whether the video frames are captured properly
    if not ret: #if vid finish repeat
        frame = cv2.VideoCapture("indian traffic1.mp4")
        continue
    if ret:  # if there is a frame continue with code
        image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
        
        cv2.imshow("image", image) 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray
        cv2.imshow("gray", gray) 
        fgmask = sub.apply(image)  # uses the background subtraction
        cv2.imshow("fgmask", fgmask) 
        
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
    
        closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    
        dilation = cv2.dilate(opening, kernel)
        cv2.imshow("dilation", dilation)
        retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # removes the shadows
        
        # creates contours
        contours, hierarchy = cv2.findContours(bins, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        minarea = 400
        # max area for contours, can be quite large for buses
        maxarea = 50000
        # vectors for the x and y locations of contour centroids in current frame
        cxx = np.zeros(len(contours))
        cyy = np.zeros(len(contours))
        #image = np.full((100,80,3), 12, np.uint8

        for i in range(len(contours)):  # cycles through all contours in current frame
            if hierarchy[0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)
                
                area = cv2.contourArea(contours[i])  # area of contour
                if minarea < area < maxarea:  # area threshold for contour
                    # calculating centroids of contours
                    cnt = contours[i]
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # x,y is top left corner and w,h is width and height
                    x, y, w, h = cv2.boundingRect(cnt)
                    # creates a rectangle around contour
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
    cv2.imshow("countours", image)
    key = cv2.waitKey(100)

    #to terminate the execution
    if key == ord('q'):
       break
    ret=True

#detroys all the windows created
cv2.destroyAllWindows();
cap.release()
