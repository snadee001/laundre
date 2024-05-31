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

def find_intersection_with_shirt(mask, start_point, slope, frame):
    x, y = start_point
    step = 5
    height, width = mask.shape
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
        lower_color = np.array([40, 20, 20])  # Example lower HSV threshold for green
        upper_color = np.array([80, 255, 255])  # Example upper HSV threshold for green

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            shirt_contour, mask = detect_shirt_by_color(frame, lower_color, upper_color)
            if shirt_contour is not None and self.start_processing:
                height, width = frame.shape[:2]

                # Find bottom left and right corners by sliding lines
                leftbottom = find_intersection_with_shirt(mask, (0, height - 1), 1, frame)
                rightbottom = find_intersection_with_shirt(mask, (width - 1, height - 1), -1, frame)

                if leftbottom and rightbottom:
                    # Draw circles at the intersection points
                    cv2.circle(frame, leftbottom, 50, (0, 0, 255), -1)
                    cv2.circle(frame, rightbottom, 50, (0, 0, 255), -1)

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