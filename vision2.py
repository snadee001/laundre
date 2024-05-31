import threading
import cv2
import numpy as np
import requests
import imutils

url = "http://10.34.146.31:8080/shot.jpg"

def fetch_image(running):
    global img_arr
    while running[0]:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)

def display_image(running):
    global img_arr
    while True:
        if img_arr is not None:
            img = cv2.imdecode(img_arr, -1)
            img = imutils.resize(img, width=600)
            
            # Convert to HSV color space
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Define the range of green color in HSV
            lower_green = np.array([40, 20, 20])
            upper_green = np.array([80, 255, 255])
            
            # Threshold the HSV image to get only green colors
            mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter and smooth contours
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Adjust this threshold as needed
                    # Approximate the contour to smooth it
                    epsilon = 0.003 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)

                    # draw circles at the left and right ends of the contour

                    # Get the leftmost and rightmost points
                    left = tuple(contour[contour[:, :, 0].argmin()][0])
                    right = tuple(contour[contour[:, :, 0].argmax()][0])

                    # Draw circles at the left and right ends
                    cv2.circle(img, left, 5, (0, 0, 255), -1)
                    cv2.circle(img, right, 5, (0, 0, 255), -1)

                    # get the bottom most point
                    bottom = tuple(contour[contour[:, :, 1].argmax()][0])

                    bottom_y = bottom[1]

                    # search over y values of the contour with close y-proximity to the bottmo_y point
                    # to find leftbottom and rightbottom points
                    # Initialize variables to store the points
                    leftbottom = bottom
                    rightbottom = bottom

                    threshold_y = 10 

                    for pt in contour:
                        x, y = pt[0][0], pt[0][1]

                        if abs(y - bottom_y) <= threshold_y:
                            if leftbottom is None or x < leftbottom[0]:
                                leftbottom = (x, y)
                            if rightbottom is None or x > rightbottom[0]:
                                rightbottom = (x, y)

                    cv2.circle(img, leftbottom, 5, (0, 0, 225), -1) 
                    cv2.circle(img, rightbottom, 5, (0, 0, 225), -1)

                    # draw a line between left and right
                    cv2.line(img, left, right, (0, 100, 0), 2)

                    midpoint = ((left[0] + right[0]) // 2, (left[1] + right[1]) // 2)
                    midpointbottom = ((leftbottom[0] + rightbottom[0]) // 2, (leftbottom[1] + rightbottom[1]) // 2)

                    # find the top point
                    top = tuple(contour[contour[:, :, 1].argmin()][0])
                    top_x, top_y = top

                    if midpoint[0] == midpointbottom[0]:
                        top_x = midpoint[0]
                    else:
                        m = (midpointbottom[1] - midpoint[1]) / (midpointbottom[0] - midpoint[0])
                        c = midpoint[1] - m * midpoint[0]

                        if m == 0:
                            top_x = midpoint[0]
                        else:
                            top_x = int((top_y - c) / m)

                    top = (top_x, top_y)
                    cv2.circle(img, top, 5, (0, 0, 225), -1)

                    cv2.line(img, top, midpointbottom, (0, 100, 0), 2)

            cv2.imshow("Android_cam", img)

            if cv2.waitKey(1) == 27:  # Escape key
                running[0] = False  # Signal to stop the fetch_image thread
                break
    cv2.destroyAllWindows()

img_arr = None
running = [True]  # Use a list to maintain a reference

thread = threading.Thread(target=fetch_image, args=(running,))
thread.start()

display_image(running)

# Make sure the thread has ended before completely exiting
thread.join()