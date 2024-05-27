import cv2
import numpy as np
import random
from threading import Event

# Function to detect shirt by color
def detect_shirt_by_color(frame, lower_color, upper_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        shirt_contour = max(contours, key=cv2.contourArea)
        return shirt_contour, mask
    return None, mask

# Function to draw lines and highlight extreme points
def draw_lines_and_highlight_extremes(frame, points, slope_threshold=0.001):
    max_x_point = points[np.argmax(points[:, 0])]
    min_x_point = points[np.argmin(points[:, 0])]
    max_y_point = points[np.argmax(points[:, 1])]
    min_y_point = points[np.argmin(points[:, 1])]

    for point in points:
        for other_point in points:
            if point is not other_point:
                slope = abs((point[1] - other_point[1]) / (point[0] - other_point[0] + 1e-6))
                if slope < slope_threshold:  # Horizontal line
                    cv2.line(frame, tuple(point), tuple(other_point), (255, 0, 0), 1)
                if slope > 1 / slope_threshold:  # Vertical line
                    cv2.line(frame, tuple(point), tuple(other_point), (0, 255, 0), 1)

    # Highlight extreme points
    cv2.circle(frame, tuple(max_x_point), 60, (0, 0, 255), -1)
    cv2.circle(frame, tuple(min_x_point), 60, (0, 0, 255), -1)
    cv2.circle(frame, tuple(max_y_point), 60, (0, 0, 255), -1)
    cv2.circle(frame, tuple(min_y_point), 60, (0, 0, 255), -1)

    return min_x_point, max_x_point, min_y_point, max_y_point

# Function to reflect a point over a line
def reflect_point_over_line(point, line_point, line_slope):
    if line_slope == 0:  # Horizontal line
        return np.array([point[0], 2 * line_point[1] - point[1]])
    if np.isinf(line_slope):  # Vertical line
        return np.array([2 * line_point[0] - point[0], point[1]])
    # For general lines
    perp_slope = -1 / line_slope
    x1, y1 = point
    x2, y2 = line_point
    d = (x1 + (y1 - y2) * perp_slope - x2) / (1 + perp_slope ** 2)
    x_reflected = 2 * x2 - x1 + 2 * d / (1 + perp_slope ** 2)
    y_reflected = 2 * y2 - y1 + 2 * d * perp_slope / (1 + perp_slope ** 2)
    return np.array([x_reflected, y_reflected])

# Function to calculate and display target points for the first fold
def first_fold(top, left, right, bottom_left, bottom_right):
    # Calculate the slope of the line passing through bottom_left and bottom_right
    bottom_slope = (bottom_right[1] - bottom_left[1]) / (bottom_right[0] - bottom_left[0] + 1e-6)
    bottom_slope = np.inf if bottom_left[0] == bottom_right[0] else bottom_slope

    # Calculate the target points by reflecting over perpendicular lines
    left_target = reflect_point_over_line(left, bottom_left, -1 / bottom_slope)
    right_target = reflect_point_over_line(right, bottom_right, -1 / bottom_slope)

    return left_target, right_target

# Function to calculate and display the target point for the second fold
def second_fold(top, bottom_left, bottom_right):
    # Calculate the midpoint of the bottom edge
    bottom_midpoint = (bottom_left + bottom_right) / 2
    return bottom_midpoint

class WebcamProcessor:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.event = Event()
        self.first_fold_targets = None
        self.second_fold_target = None

    def process_stream(self):
        lower_color = np.array([35, 100, 100])  # Example lower HSV threshold for green
        upper_color = np.array([85, 255, 255])  # Example upper HSV threshold for green
        first_fold_completed = False
        second_fold_completed = False

        while self.cap.isOpened():
            ret, color_frame = self.cap.read()
            if not ret:
                break

            shirt_contour, mask = detect_shirt_by_color(color_frame, lower_color, upper_color)

            if shirt_contour is not None:
                # Draw contours and process points
                cv2.drawContours(color_frame, [shirt_contour], -1, (0, 255, 0), 2)
                points = np.array(random.sample(list(shirt_contour.reshape(-1, 2)), 30))
                min_x_point, max_x_point, min_y_point, max_y_point = draw_lines_and_highlight_extremes(color_frame, points)

                key = cv2.waitKey(1)
                first_fold_completed = False
                if key == ord('c') and not first_fold_completed:
                    # Calculate and display target points for the first fold when 'c' is pressed
                    left_target, right_target = first_fold(max_y_point, min_x_point, max_x_point, min_y_point, max_y_point)
                    self.first_fold_targets = (left_target, right_target)
                    first_fold_completed = True

                if first_fold_completed:
                    # Draw the first fold target points
                    for target in self.first_fold_targets:
                        cv2.circle(color_frame, tuple(target.astype(int)), 60, (0, 255, 255), -1)

                    # Print statements when points come within a range of targets
                    if np.linalg.norm(min_x_point - self.first_fold_targets[0]) < 50:
                        print("Min X point is close to its target for first fold")
                    if np.linalg.norm(max_x_point - self.first_fold_targets[1]) < 50:
                        print("Max X point is close to its target for first fold")

                    # If first fold conditions are met, calculate target for second fold
                    if not second_fold_completed and \
                       np.linalg.norm(min_x_point - self.first_fold_targets[0]) < 50 and \
                       np.linalg.norm(max_x_point - self.first_fold_targets[1]) < 50:
                        top_target = second_fold(max_y_point, min_y_point, max_y_point)
                        self.second_fold_target = top_target
                        second_fold_completed = True

                if second_fold_completed:
                    # Draw the second fold target point
                    cv2.circle(color_frame, tuple(self.second_fold_target.astype(int)), 60, (255, 0, 255), -1)

                    # Print statements when the top point comes within a range of the target
                    if np.linalg.norm(max_y_point - self.second_fold_target) < 50:
                        print("Top point is close to its target for second fold")

            cv2.imshow('Color Segmentation Mask', mask)
            cv2.imshow('Color Frame with Contours and Lines', color_frame)

            if cv2.waitKey(1) == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    processor = WebcamProcessor()
    processor.process_stream()