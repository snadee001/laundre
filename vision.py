import cv2
import numpy as np
from threading import Event


# Function to detect shirt by color
def detect_by_color(frame, lower_color, upper_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    return mask

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

def find_right_sleeve_top(mask):
    width, height = mask.shape
    mask_list = [(x, y) for x in range(width) for y in range(height) if mask[x][y]]
    mask_list_edited = [y for (x, y) in mask_list]
    mask_dict = {mask_list[i]: mask_list_edited[i] for i in range(len(mask_list))}
    res = max(mask_dict, key=mask_dict.get)
    return res

def find_left_sleeve_top(mask):
    width, height = mask.shape
    mask_list = [(x, y) for x in range(width) for y in range(height) if mask[x][y]]
    mask_list_edited = [y for (x, y) in mask_list]
    mask_dict = {mask_list[i]: mask_list_edited[i] for i in range(len(mask_list))}
    res = min(mask_dict, key=mask_dict.get)
    return res

def find_sleeve(mask, bottom, sleeve_top):
    width, _ = mask.shape
    inner_y = int(bottom[1]+(sleeve_top[1]-bottom[1])*0.05)
    line = [x for x in range(width) if mask[x][inner_y]]
    x_mid = int((min(line)+max(line))/2)
    sleeve_mid_outer = (x_mid, sleeve_top[1])
    sleeve_mid_inner = (x_mid, bottom[1])
    return sleeve_mid_outer, sleeve_mid_inner

def projection(green_mask, markers):
    width, height = green_mask.shape
    #dest_markers = np.float32([[0.1*height,0.05*width], [0.1*height, 0.9*width], [0.9*height, 0.9*width], [0.9*height, 0.05*width]])
    dest_markers = np.float32([[0.0*height,0.0*width], [0.0*height, 1.0*width], [1.0*height, 1.0*width], [1.0*height, 0.0*width]])
    M = cv2.getPerspectiveTransform(np.float32(markers), dest_markers)
    return cv2.warpPerspective(green_mask, M, (height, width))

def barycentric_weights(x1, y1, x2, y2, x3, y3, x4, y4, xn, yn):
    def area(x1, y1, x2, y2, x3, y3):
        return 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    
    A = area(x1, y1, x2, y2, x3, y3) + area(x1, y1, x3, y3, x4, y4)
    
    A1 = area(xn, yn, x2, y2, x3, y3) + area(xn, yn, x3, y3, x4, y4)
    A2 = area(x1, y1, xn, yn, x3, y3) + area(x1, y1, x3, y3, x4, y4)
    A3 = area(x1, y1, x2, y2, xn, yn) + area(x1, y1, xn, yn, x4, y4)
    A4 = area(x1, y1, x2, y2, x3, y3) + area(x1, y1, x3, y3, xn, yn)
    
    lambda1 = A1 / A
    lambda2 = A2 / A
    lambda3 = A3 / A
    lambda4 = A4 / A
    
    return np.array([lambda1, lambda2, lambda3, lambda4])

def convert_to_world_coords(mask, point):
    width, height = mask.shape
    x1, y1 = 0, height
    x2, y2 = 0.9*width, height
    x3, y3 = 0.9*width, 0
    x4, y4 = 0, 0

    physical_points = np.array([[0.252, -0.527], [0.252, 0.463], [0.925, 0.463], [0.925, -0.527]])
    weights = barycentric_weights(x1, y1, x2, y2, x3, y3, x4, y4, point[0], point[1])

    world_coord = np.zeros(2)
    for i in range(4):
        world_coord += physical_points[i]*weights[i]
    
    return world_coord

class WebcamProcessor:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.event = Event()
        self.start_processing = False

    def process_stream(self):
        lower_color_red = np.array([0, 150, 150])  # Example lower HSV threshold for red
        upper_color_red = np.array([25, 220, 255])  # Example upper HSV threshold for red

        lower_color = np.array([35, 100, 100])  # Example lower HSV threshold for green
        upper_color = np.array([85, 255, 255])  # Example upper HSV threshold for green

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = frame[400:1380, :1000]

            if self.start_processing:
                mask_red = detect_by_color(frame, lower_color_red, upper_color_red)

                leftbottom_red = find_left_bottom_corner(mask_red)
                rightbottom_red = find_right_bottom_corner(mask_red)
                lefttop_red = find_left_top_corner(mask_red)
                righttop_red = find_right_top_corner(mask_red)
                points_red = [lefttop_red, righttop_red, rightbottom_red, leftbottom_red]

                mask = detect_by_color(frame, lower_color, upper_color)
                mask = projection(mask, points_red)

                leftbottom = find_left_bottom_corner(mask)
                rightbottom = find_right_bottom_corner(mask)

                left_sleeve_top = find_left_sleeve_top(mask)
                right_sleeve_top = find_right_sleeve_top(mask)
                left_sleeve_outer, left_sleeve_inner = find_sleeve(mask, leftbottom, left_sleeve_top)
                right_sleeve_outer, right_sleeve_inner = find_sleeve(mask, rightbottom, right_sleeve_top)

                shirt_height = leftbottom[0] - left_sleeve_top[0]
                shirt_width = rightbottom[1]-leftbottom[1]
                leftbottom_target = (int(leftbottom[0]-shirt_height/5), int(leftbottom[1]+0.25*shirt_width))
                rightbottom_target = (int(rightbottom[0]-shirt_height/5), int(rightbottom[1]-0.25*shirt_width))

                midbottom = (int((leftbottom[0]+rightbottom[0])/2), int((leftbottom[1]+rightbottom[1])/2))
                mid2 = (midbottom[0]-shirt_height/3, midbottom[1])
                mid3 = (midbottom[0]-2*shirt_height/3, midbottom[1])

                third = (left_sleeve_inner[0], left_sleeve_inner[1]+shirt_width/3)
                third2 = (left_sleeve_inner[0], left_sleeve_inner[1]+2*shirt_width/3)

                points = [left_sleeve_outer, left_sleeve_inner, right_sleeve_outer, right_sleeve_inner, \
                          leftbottom, leftbottom_target, rightbottom, rightbottom_target, midbottom, \
                            mid2, mid2, mid3, left_sleeve_inner, third, third, third2]

                for point in points:
                    cv2.circle(frame, (int(point[1]), int(point[0])), 25, (0, 0, 255), -1)
                    cv2.circle(mask, (int(point[1]), int(point[0])), 25, (0, 0, 255), -1)

                # Display both the original frame and the mask
                combined_image = np.hstack((frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))
            else:
                combined_image = frame

            cv2.imshow('Webcam Frame and Mask', combined_image)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.start_processing = True
            elif key == ord('a'): #Write to file
                with open("points.txt", "w") as file:
                    for point in points:
                        #change to robot base frame w/ barycentric first
                        point = convert_to_world_coords(mask, point)
                        file.write(f"{point[0]} {point[1]}\n")

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    processor = WebcamProcessor()
    processor.process_stream()