import cv2
import numpy as np
from threading import Event

# Copied from vision.py
def detect_by_color(frame, lower_color, upper_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    return mask

# Copied from vision.py
def find_right_bottom_corner(mask):
    width, height = mask.shape
    mask_list = [(x, y) for x in range(width) for y in range(height) if mask[x][y]]
    mask_list_edited = [x + y for (x, y) in mask_list]
    mask_dict = {mask_list[i]: mask_list_edited[i] for i in range(len(mask_list))}
    res = max(mask_dict, key=mask_dict.get)
    return res

def find_right_top_corner(mask):
    width, height = mask.shape
    mask_list = [(x, y) for x in range(width) for y in range(height) if mask[x][y]]
    mask_list_edited = [y - x for (x, y) in mask_list]
    mask_dict = {mask_list[i]: mask_list_edited[i] for i in range(len(mask_list))}
    res = max(mask_dict, key=mask_dict.get)
    return res

# Copied from vision.py
def find_left_bottom_corner(mask):
    width, height = mask.shape
    mask_list = [(x, y) for x in range(width) for y in range(height) if mask[x][y]]
    mask_list_edited = [x - y for (x, y) in mask_list]
    mask_dict = {mask_list[i]: mask_list_edited[i] for i in range(len(mask_list))}
    res = max(mask_dict, key=mask_dict.get)
    return res

def find_left_top_corner(mask):
    width, height = mask.shape
    mask_list = [(x, y) for x in range(width) for y in range(height) if mask[x][y]]
    mask_list_edited = [x + y for (x, y) in mask_list]
    mask_dict = {mask_list[i]: mask_list_edited[i] for i in range(len(mask_list))}
    res = min(mask_dict, key=mask_dict.get)
    return res

# Modified from vision.py
def process_single_frame(self, frame):
    lower_color_red = np.array([0, 150, 150])  # Example lower HSV threshold for red
    upper_color_red = np.array([25, 230, 255])  # Example upper HSV threshold for red

    mask_red = detect_by_color(frame, lower_color_red, upper_color_red)
    points_red = {}

    leftbottom_red = find_left_bottom_corner(mask_red)
    points_red["leftbottom_red"] = leftbottom_red

    rightbottom_red = find_right_bottom_corner(mask_red)
    points_red["rightbottom_red"] = rightbottom_red

    lefttop_red = find_left_top_corner(mask_red)
    points_red["lefttop_red"] = lefttop_red

    righttop_red = find_right_top_corner(mask_red)
    points_red["righttop_red"] = righttop_red

    for point_red in points_red.values():
        cv2.circle(frame, (int(point_red[1]), int(point_red[0])), 10, (255, 0, 0), -1)

    with open("points_red.txt", "w") as file:
        file.write(f"{lefttop_red[0]} {lefttop_red[1]}\n")
        file.write(f"{righttop_red[0]} {righttop_red[1]}\n")
        file.write(f"{rightbottom_red[0]} {rightbottom_red[1]}\n")
        file.write(f"{leftbottom_red[0]} {leftbottom_red[1]}\n")

    # ... perspective transform
        
    # ... green threshold and the rest of the code
        
    combined_image = np.hstack(np.hstack((frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))), cv2.cvtColor(mask_red, cv2.COLOR_GRAY2BGR))
    cv2.imshow('Webcam Frame and Mask', combined_image)

def process_stream(self):

    lower_color_red = np.array([0, 150, 150])  # Example lower HSV threshold for red
    upper_color_red = np.array([25, 230, 255])  # Example upper HSV threshold for red

    # This part is already in vision.py
    while self.cap.isOpened():
        ret, frame = self.cap.read()
        if not ret:
            break
        
        # This part is new
        mask_red = detect_by_color(frame, lower_color_red, upper_color_red)
        points_red = []

        leftbottom = find_left_bottom_corner(mask_red)
        points_red.append(leftbottom)

        rightbottom = find_right_bottom_corner(mask_red)
        points_red.append(rightbottom)

        lefttop = find_left_top_corner(mask_red)
        points_red.append(lefttop)

        righttop = find_right_top_corner(mask_red)
        points_red.append(righttop)

        for point_red in points_red:
            cv2.circle(frame, (int(point_red[1]), int(point_red[0])), 50, (255, 0, 0), -1)

        # ... perspective transform and the rest of the code
            
        # ... green threshold

        combined_image = np.hstack(np.hstack((frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))), cv2.cvtColor(mask_red, cv2.COLOR_GRAY2BGR))
        cv2.imshow('Webcam Frame and Mask', combined_image)
        
        
