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

def detect_armpit_corners(contour, frame):
    # Create an empty image to draw the contour
    contour_img = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_img, [contour], -1, 255, 1)

    # Apply Canny edge detection
    edges = cv2.Canny(contour_img, 100, 200)

    # Find contours on the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the shirt contour
    if contours:
        shirt_contour = max(contours, key=cv2.contourArea)

        # Approximate the contour to get a simplified version of it
        epsilon = 0.02 * cv2.arcLength(shirt_contour, True)
        approx = cv2.approxPolyDP(shirt_contour, epsilon, True)

        # Draw the approximated contour
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

        # Detect corners in the approximated contour
        corners = []
        for i in range(len(approx)):
            pt1 = approx[i][0]
            pt2 = approx[(i + 1) % len(approx)][0]
            pt3 = approx[(i + 2) % len(approx)][0]

            angle = np.abs(np.arctan2(pt3[1] - pt2[1], pt3[0] - pt2[0]) - np.arctan2(pt1[1] - pt2[1], pt1[0] - pt2[0]))
            if angle > np.pi:
                angle = 2 * np.pi - angle

            # Assuming that the armpit corners have sharp angles
            if angle > np.pi / 4 and angle < 3 * np.pi / 4:
                corners.append(pt2)
                cv2.circle(frame, tuple(pt2), 10, (255, 0, 0), -1)

        return corners
    return []

def draw_perpendicular_line(mask, point, slope, frame):
    x, y = point
    height, width = mask.shape
    step = 5
    perpendicular_points = []

    # Calculate the perpendicular slope
    perpendicular_slope = -1 / slope

    # Check along the perpendicular line
    for step_multiplier in range(-max(height, width) // step, max(height, width) // step):
        x_curr = x - step_multiplier * step
        y_curr = int(y + step_multiplier * step * perpendicular_slope)
        if 0 <= x_curr < width and 0 <= y_curr < height:
            perpendicular_points.append((x_curr, y_curr))
            cv2.circle(frame, (x_curr, y_curr), 2, (0, 255, 0), -1)  # Drawing perpendicular line points

    return perpendicular_points

def detect_slope_changes(contour, threshold=0.1):
    def calculate_slope(pt1, pt2):
        if pt2[0] - pt1[0] == 0:
            return float('inf')  # Vertical line
        return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])

    slopes = []
    changes = []

    for i in range(len(contour) - 1):
        pt1 = contour[i][0]
        pt2 = contour[i + 1][0]
        slope = calculate_slope(pt1, pt2)
        slopes.append(slope)

    for i in range(1, len(slopes)):
        if abs(slopes[i] - slopes[i - 1]) > threshold:
            changes.append(contour[i][0])
    
    return changes

def detect_slope_changes_with_harris(contour, frame, blockSize=5, ksize=7, k=0.1):
    # Create an empty image to draw the contour
    contour_img = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_img, [contour], -1, 255, 1)

    # Detect corners using Harris Corner Detection
    dst = cv2.cornerHarris(contour_img, blockSize, ksize, k)
    dst = cv2.dilate(dst, None)

    # Threshold for detecting strong corners
    corners = np.argwhere(dst > 0.5 * dst.max())

    # Draw detected corners on the original frame
    print(len(corners))
    for corner in corners:
        y, x = corner
        cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)

    return corners

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

    
    step = 5
    
    white_points_threshold = 3

    while y >= 0:
        count_white_points = 0
        x_curr = x
        y_curr = y
        points_on_line = []

        for _ in range(min(height, width)):
            if 0 <= x_curr < width and 0 <= y_curr < height:
                points_on_line.append((x_curr, y_curr))
                if mask[y_curr, x_curr] > 0:
                    count_white_points += 1

            x_curr += slope * step
            y_curr -= step

        # Draw the diagonal line being checked
        for point in points_on_line:
            cv2.circle(frame, point, 2, (255, 0, 0), -1)  # Drawing line points

        # Draw and check along the perpendicular line
        perpendicular_points = draw_perpendicular_line(mask, (x, y), slope, frame)
        count_white_points = sum(1 for pt in perpendicular_points if mask[pt[1], pt[0]] > 0)

        if count_white_points >= white_points_threshold:
            # Find the average of the three points satisfying the condition
            corner_point = tuple(np.mean(perpendicular_points[:3], axis=0).astype(int))
            cv2.circle(frame, corner_point, 50, (0, 0, 255), -1)  # Drawing corner point
            return corner_point

        y -= step

    return None

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
                cv2.drawContours(frame, [shirt_contour], -1, (255, 0, 0), 2)
                #dst = cv2.cvtColor(frame.cv2.COLORBGR2GRAY)
                # width, height = mask.shape
                # dst = cv2.cornerHarris(mask, 2, 3, 0.04)
                # dst = [(x, y) for x in range(width) for y in range(height) if mask[x][y]]
                # print(dst)
                # dst = cv2.dilate(dst, None)
                # mask[dst>0.01*dst.max()] = [0, 0, 255]
                # cv2.imshow('dst', mask)
                # points = detect_slope_changes(shirt_contour)
                # height, width = frame.shape[:2]

                corners = detect_armpit_corners(shirt_contour, frame)
                for c in corners:
                    cv2.circle(frame, c, 50, (0, 0, 255), -1)

                # Find bottom left and right corners by sliding lines
                leftbottom = find_left_bottom_corner(mask)
                rightbottom = find_right_bottom_corner(mask)

                if leftbottom and rightbottom:
                    pass
                    # Draw circles at the intersection points
                    #cv2.circle(frame, leftbottom, 50, (0, 0, 255), -1)
                    #cv2.circle(frame, rightbottom, 50, (0, 0, 255), -1)

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