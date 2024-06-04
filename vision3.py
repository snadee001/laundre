import cv2
import numpy as np
from threading import Event

# Function to detect shirt by color
def detect_shirt_by_color(frame, lower_color, upper_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        shirt_contour = max(contours, key=cv2.contourArea)
        return shirt_contour, mask
    return None, mask

def find_right_bottom_corner(mask):
    width, height = mask.shape
    mask_list = [(x, y) for x in range(width) for y in range(height) if mask[x][y]]
    mask_list_edited = [(height - y) + (width - x) for (x, y) in mask_list]
    mask_dict = {mask_list[i]: mask_list_edited[i] for i in range(len(mask_list))}
    res = min(mask_dict, key=mask_dict.get)
    return (res[1], res[0])

def find_left_bottom_corner(mask):
    width, height = mask.shape
    mask_list = [(x, y) for x in range(width) for y in range(height) if mask[x][y]]
    mask_list_edited = [(height - y) + x for (x, y) in mask_list]
    mask_dict = {mask_list[i]: mask_list_edited[i] for i in range(len(mask_list))}
    res = max(mask_dict, key=mask_dict.get)
    return (res[1], res[0])

def find_right_sleeve_top(mask):
    width, height = mask.shape
    mask_list = [(x, y) for x in range(width) for y in range(height) if mask[x][y]]
    mask_list_edited = [x for (x, y) in mask_list]
    mask_dict = {mask_list[i]: mask_list_edited[i] for i in range(len(mask_list))}
    res = max(mask_dict, key=mask_dict.get)
    return (res[1], res[0])

def find_left_sleeve_top(mask):
    width, height = mask.shape
    mask_list = [(x, y) for x in range(width) for y in range(height) if mask[x][y]]
    mask_list_edited = [x for (x, y) in mask_list]
    mask_dict = {mask_list[i]: mask_list_edited[i] for i in range(len(mask_list))}
    res = min(mask_dict, key=mask_dict.get)
    return (res[1], res[0])

def find_sleeve(mask, bottom, sleeve_top):
    width, height = mask.shape
    inner_x = bottom[0]+(sleeve_top[0]-bottom[0])*0.05
    line = [y for y in range(height) if mask[inner_x][y]]
    y_mid = (min(line)+max(line))/2
    sleeve_mid_outer = (sleeve_top[0], y_mid)
    sleeve_mid_inner = (bottom[0], y_mid)
    return sleeve_mid_outer, sleeve_mid_inner

class WebcamProcessor:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.event = Event()
        self.start_processing = False

    def process_stream(self):
        lower_color = np.array([35, 100, 100])  # Example lower HSV threshold for green
        upper_color = np.array([85, 255, 255])  # Example upper HSV threshold for green

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            shirt_contour, mask = detect_shirt_by_color(frame, lower_color, upper_color)
            if shirt_contour is not None and self.start_processing:
                leftbottom = find_left_bottom_corner(mask)
                rightbottom = find_right_bottom_corner(mask)
                left_sleeve_top = find_left_sleeve_top(mask)
                right_sleeve_top = find_right_sleeve_top(mask)
                left_sleeve_outer, left_sleeve_inner = find_sleeve(mask, leftbottom, left_sleeve_top)
                right_sleeve_outer, right_sleeve_inner = find_sleeve(mask, rightbottom, right_sleeve_top)

                shirt_height = left_sleeve_top[1]-leftbottom[1]
                shirt_width = rightbottom[0]-leftbottom[0]
                leftbottom_target = (leftbottom[0]+0.25*shirt_width, leftbottom[1]-shirt_height/6)
                rightbottom_target = (rightbottom[0]-0.25*shirt_width, rightbottom[1]-shirt_height/6)
                midbottom = ((leftbottom[0]+rightbottom[0])/2, (leftbottom[1]+rightbottom[1])/2)
                mid2 = (midbottom[0], midbottom[1]-shirt_height/3)
                mid3 = (midbottom[0], midbottom[1]-2*shirt_height/3)
                third = (left_sleeve_inner[0]+shirt_width/3, left_sleeve_inner[1])
                third2 = (left_sleeve_inner[0]+2*shirt_width/3, left_sleeve_inner[1])

                for point in [leftbottom, rightbottom, left_sleeve_outer, left_sleeve_inner, right_sleeve_outer, right_sleeve_inner, leftbottom_target, rightbottom_target, midbottom, mid2, mid3, third, third2]:
                    cv2.circle(frame, point, 50, (0, 0, 255), -1)
                    
            # Display both the original frame and the mask
            combined_image = np.hstack((frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))
            cv2.imshow('Webcam Frame and Mask', combined_image)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.start_processing = True

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    processor = WebcamProcessor()
    processor.process_stream()