import numpy as np

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

if __name__ == "__main__":
    # Input 5 points
    print("Enter the coordinates for 5 points (top, left, right, bottom_left, bottom_right):")
    top = np.array(list(map(int, input("Top point (x y): ").split())))
    left = np.array(list(map(int, input("Left point (x y): ").split())))
    right = np.array(list(map(int, input("Right point (x y): ").split())))
    bottom_left = np.array(list(map(int, input("Bottom-left point (x y): ").split())))
    bottom_right = np.array(list(map(int, input("Bottom-right point (x y): ").split())))

    # Calculate target points for the first fold
    left_target, right_target = first_fold(top, left, right, bottom_left, bottom_right)

    # Print target points for the first fold
    print(f"First Fold - Left target point: {left_target}")
    print(f"First Fold - Right target point: {right_target}")

    # Calculate the target point for the second fold
    top_target = second_fold(top, bottom_left, bottom_right)

    # Print target point for the second fold
    print(f"Second Fold - Top target point: {top_target}")