import numpy as np

# Function to calculate target points for all folds
def get_target_points(top_left, top_right, middle_left, middle_right, bottom_left, bottom_right):
    top_center_1 = top_left + (top_right - top_left) / 3
    top_center_2 = top_right - (top_right - top_left) / 3
    middle_center_1 = middle_left + (middle_right - middle_left) / 3
    middle_center_2 = middle_right - (middle_right - middle_left) / 3
    middle_edge_1 = (middle_left + bottom_left) / 2
    middle_edge_2 = (middle_right + bottom_right) / 2
    midpoint_1 = (middle_edge_1 + middle_edge_2) / 2
    midpoint_2 = (bottom_left + bottom_right) / 2
    ratio = (((midpoint_1[0] - midpoint_2[0]) ** 2 + (midpoint_1[1] - midpoint_2[1]) ** 2) ** 0.5) / (((bottom_left[0] - bottom_right[0]) ** 2 + (bottom_left[1] - bottom_right[1]) ** 2) ** 0.5)
    bottom_center_1 = bottom_left + ratio * (bottom_right - bottom_left)
    bottom_center_2 = bottom_right - ratio * (bottom_right - bottom_left)

    return top_center_1, top_center_2, middle_center_1, middle_center_2, middle_edge_1, middle_edge_2, bottom_center_1, bottom_center_2

if __name__ == "__main__":
    # Input 6 points
    print("Enter the coordinates for 6 points (top left, top right, middle left, middle right, bottom left, bottom right):")
    top_left = np.array(list(map(int, input("Top left point (x y): ").split())))
    top_right = np.array(list(map(int, input("Top right point (x y): ").split())))
    middle_left = np.array(list(map(int, input("Middle left point (x y): ").split())))
    middle_right = np.array(list(map(int, input("Middle right point (x y): ").split())))
    bottom_left = np.array(list(map(int, input("Bottom left point (x y): ").split())))
    bottom_right = np.array(list(map(int, input("Bottom right point (x y): ").split())))

    # Calculate target points for all folds
    top_center_1, top_center_2, middle_center_1, middle_center_2, middle_edge_1, middle_edge_2, bottom_center_1, bottom_center_2 = get_target_points(top_left, top_right, middle_left, middle_right, bottom_left, bottom_right)

    # Print target points
    print(f"First Fold - Left Sleeve Center: {(top_left + middle_left) / 2}")

    print(f"Second Fold - Right Sleeve Center: {(top_right + middle_right) / 2}")

    print(f"Third Fold - Bottom Left Corner Center: {(middle_edge_1 + bottom_center_1) / 2}")

    print(f"Fourth Fold - Bottom Right Corner Center: {(middle_edge_2 + bottom_center_2) / 2}")

    print(f"Fifth Fold - Bottom Flap Center: {(middle_edge_1 + middle_edge_2) / 2}")

    print(f"Sixth Fold - Middle Flap Center: {(middle_left + middle_right) / 2}")

    print(f"Seventh Fold - Left Center: {(top_center_1 + middle_center_1) / 2}")

    print(f"Eighth Fold - Center Center: {(top_center_2 + middle_center_2) / 2}")