# import cv2
# import numpy as np
# import json

# def detect_circles(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (9, 9), 1)
#     circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.5, minDist=20,
#                                param1=50, param2=30, minRadius=25, maxRadius=28)
#     if circles is not None:
#         circles = np.round(circles[0, :]).astype("int")
#         return circles
#     return []

# def group_circles(circles, y_threshold=15):
#     # Sort circles by y-coordinate (top to bottom), then x-coordinate (left to right)
#     circles = sorted(circles, key=lambda c: (c[1], c[0]))
    
#     # Group circles into rows based on y-axis proximity
#     rows = []
#     current_row = []
    
#     for circle in circles:
#         if not current_row:
#             current_row.append(circle)
#         else:
#             last_y = current_row[-1][1]
#             if abs(circle[1] - last_y) <= y_threshold:
#                 current_row.append(circle)
#             else:
#                 rows.append(current_row)
#                 current_row = [circle]
#     if current_row:
#         rows.append(current_row)
    
#     # Split each row into groups of 4 circles (left to right)
#     all_groups = []
#     for row in rows:
#         sorted_row = sorted(row, key=lambda c: c[0])  # Ensure left-to-right order
#         for i in range(0, len(sorted_row), 4):
#             group = sorted_row[i:i+4]
#             if len(group) == 4:  # Only keep full groups
#                 all_groups.append(group)
        
#     return all_groups

# # def group_circles(circles, y_threshold=15):
# #     # Sort circles by y-coordinate first (top to bottom), then by x (left to right)
# #     circles = sorted(circles, key=lambda x: (x[1], x[0]))
    
# #     grouped = []
# #     column_candidates = []
    
# #     for circle in circles:
# #         if not column_candidates:
# #             column_candidates.append(circle)
# #         else:
# #             last_circle = column_candidates[-1]
# #             if abs(circle[1] - last_circle[1]) <= y_threshold:
# #                 column_candidates.append(circle)
# #             else:
# #                 grouped.append(sorted(column_candidates, key=lambda x: x[0]))  # Sort column left to right
# #                 column_candidates = [circle]
    
# #     if column_candidates:
# #         grouped.append(sorted(column_candidates, key=lambda x: x[0]))
    
# #     # New array to store groups of circles in top to bottom approach
# #     top_to_bottom_groups = []
# #     for column in grouped:
# #         sorted_column = sorted(column, key=lambda x: x[1])  # Sort each column by y-coordinate (top to bottom)
# #         top_to_bottom_groups.append(sorted_column)
    
# #     # Create groups with correct ordering (top to bottom across all columns)
# #     all_groups = []
# #     max_items_per_column = max(len(column) for column in top_to_bottom_groups) if top_to_bottom_groups else 0
    
# #     # For each position from top to bottom
# #     for i in range(0, max_items_per_column, 4):
# #         # Go through each column
# #         for column in top_to_bottom_groups:
# #             # If this column has enough items for a complete group at this position
# #             if i + 3 < len(column):
# #                 group = column[i:i+4]
# #                 all_groups.append(group)
    
# #     return all_groups

# def process_row(image, circles):
#     results = {}
#     for idx, (x, y, r) in enumerate(circles):
#         cropped = image[y - r:y + r, x - r:x + r]
#         gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
#         _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
#         pixel_count = np.sum(binary == 255)
#         results[idx] = pixel_count
#     return results

# def draw_debug_info(image, grouped_circles):
#     debug_image = image.copy()
#     for col_idx, column in enumerate(grouped_circles):
#         x_min = min(circle[0] - circle[2] for circle in column)
#         y_min = min(circle[1] - circle[2] for circle in column)
#         x_max = max(circle[0] + circle[2] for circle in column)
#         y_max = max(circle[1] + circle[2] for circle in column)
#         cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#         cv2.putText(debug_image, f"Group {col_idx}", (x_min, y_min - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#         for circle_idx, (x, y, r) in enumerate(column):
#             cv2.circle(debug_image, (x, y), r, (0, 0, 255), 2)
#             cv2.putText(debug_image, str(circle_idx), (x - 10, y + 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#     return debug_image

# def main(image_path, debug=False):
#     image = cv2.imread(image_path)
#     circles = detect_circles(image)
    
#     if debug:
#         print(f"Total circles detected: {len(circles)}")
#         debug_image = image.copy()
#         for (x, y, r) in circles:
#             cv2.circle(debug_image, (x, y), r, (0, 255, 0), 2)
#         cv2.imshow("Detected Circles", debug_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
    
#     grouped_circles = group_circles(circles, y_threshold=15)
    
#     if debug:
#         print(f"Total groups detected: {len(grouped_circles)}")
#         for i, group in enumerate(grouped_circles):
#             print(f"Group {i}: {group}")
    
#     if debug and grouped_circles:
#         debug_image = draw_debug_info(image, grouped_circles)
#         height, width = debug_image.shape[:2]
#         debug_image = cv2.resize(debug_image, (width // 2, height // 2))
#         cv2.imshow("Grouped Circles", debug_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
    
#     final_results = {}
#     for col_idx, column in enumerate(grouped_circles):
#         row_results = process_row(image, column)
#         max_index = max(row_results, key=row_results.get)
#         final_results[col_idx] = max_index
    
#     print(json.dumps(final_results, indent=4))

# if __name__ == "__main__":
#     image_path = "bubble_sheet.jpg"
#     main(image_path, debug=True)

import cv2
import numpy as np
import json
import math

def detect_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 1)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.5, minDist=20,
                               param1=50, param2=30, minRadius=25, maxRadius=28)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles
    return []

def group_circles(circles, y_threshold=15):
    circles = sorted(circles, key=lambda c: (c[1], c[0]))
    rows = []
    current_row = []
    for circle in circles:
        if not current_row:
            current_row.append(circle)
        else:
            last_y = current_row[-1][1]
            if abs(circle[1] - last_y) <= y_threshold:
                current_row.append(circle)
            else:
                rows.append(current_row)
                current_row = [circle]
    if current_row:
        rows.append(current_row)
    
    all_groups = []
    for row in rows:
        sorted_row = sorted(row, key=lambda c: c[0])
        for i in range(0, len(sorted_row), 4):
            group = sorted_row[i:i+4]
            if len(group) == 4:
                all_groups.append(group)
    return all_groups

def group_circles_with_item(circles, y_threshold=15):
    groups = group_circles(circles, y_threshold)
    result = []
    even_item = 1
    for idx, group in enumerate(groups):
        if idx % 2 == 0:
            result.append({"item": even_item, "group": group})
            even_item += 1
    odd_item = 26
    for idx, group in enumerate(groups):
        if idx % 2 == 1:
            result.append({"item": odd_item, "group": group})
            odd_item += 1
    return result

def process_row(image, circles):
    """
    Process each circle in the group:
    - Calculate the white pixel percentage within the circle area.
    - Only if exactly one circle has a percentage >=85%, return its corresponding letter.
    - Otherwise, return None.
    """
    mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    percentages = {}
    
    # Calculate percentage for each circle in the group.
    for idx, (x, y, r) in enumerate(circles):
        # Crop the circle region
        cropped = image[y - r:y + r, x - r:x + r]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        pixel_count = np.sum(binary == 255)
        circle_area = math.pi * (r ** 2)
        percentage = (pixel_count / circle_area) * 100 if circle_area > 0 else 0
        percentages[mapping.get(idx, idx)] = percentage
    
    # Find circles that meet the 85% threshold.
    qualifiers = {key: perc for key, perc in percentages.items() if perc >= 85}
    
    # Return the letter only if exactly one circle qualifies.
    if len(qualifiers) == 1:
        return list(qualifiers.keys())[0]
    else:
        return None

def draw_debug_info_with_item(image, grouped_objects):
    debug_image = image.copy()
    for group_obj in grouped_objects:
        item_number = group_obj["item"]
        group = group_obj["group"]
        x_min = min(circle[0] - circle[2] for circle in group)
        y_min = min(circle[1] - circle[2] for circle in group)
        x_max = max(circle[0] + circle[2] for circle in group)
        y_max = max(circle[1] + circle[2] for circle in group)
        cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(debug_image, f"Item {item_number}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        for circle_idx, (x, y, r) in enumerate(group):
            cv2.circle(debug_image, (x, y), r, (0, 0, 255), 2)
            cv2.putText(debug_image, str(circle_idx), (x - 10, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return debug_image

# --- New Function to Show Cropped Group with Percentage ---
def show_cropped_groups_with_percentage(image, grouped_circles):
    """
    For each group, crop the group area from the image and overlay the
    percentage of white pixels in each detected circle.
    """
    for idx, group_obj in enumerate(grouped_circles):
        group = group_obj["group"]
        x_min = min(circle[0] - circle[2] for circle in group)
        y_min = min(circle[1] - circle[2] for circle in group)
        x_max = max(circle[0] + circle[2] for circle in group)
        y_max = max(circle[1] + circle[2] for circle in group)
        
        cropped_group = image[y_min:y_max, x_min:x_max].copy()
        
        for (x, y, r) in group:
            x_rel = x - x_min
            y_rel = y - y_min

            circle_crop = image[y - r:y + r, x - r:x + r]
            if circle_crop.size == 0:
                continue
            gray = cv2.cvtColor(circle_crop, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            pixel_count = np.sum(binary == 255)
            circle_area = math.pi * (r ** 2)
            percentage = (pixel_count / circle_area) * 100 if circle_area > 0 else 0

            cv2.circle(cropped_group, (x_rel, y_rel), r, (0, 255, 0), 2)
            cv2.putText(cropped_group, f"{percentage:.1f}%", (x_rel - 10, y_rel),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.imshow(f"Cropped Group {idx+1}", cropped_group)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# --- End New Function ---

def main(image_path, debug=False):
    image = cv2.imread(image_path)
    circles = detect_circles(image)
    
    if debug:
        print(f"Total circles detected: {len(circles)}")
        debug_image = image.copy()
        for (x, y, r) in circles:
            cv2.circle(debug_image, (x, y), r, (0, 255, 0), 2)
        cv2.imshow("Detected Circles", debug_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    grouped_circles = group_circles_with_item(circles, y_threshold=15)
    
    if debug:
        print(f"Total groups detected: {len(grouped_circles)}")
        for i, group_obj in enumerate(grouped_circles):
            print(f"Group {i}: {group_obj}")
        
        debug_image = draw_debug_info_with_item(image, grouped_circles)
        height, width = debug_image.shape[:2]
        debug_image = cv2.resize(debug_image, (width // 2, height // 2))
        cv2.imshow("Grouped Circles", debug_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        show_cropped_groups_with_percentage(image, grouped_circles)
    
    final_results = {}
    for group_obj in grouped_circles:
        letter = process_row(image, group_obj["group"])
        # If process_row returns None, output null for that group
        final_results[group_obj["item"]] = letter if letter is not None else None
    
    print(json.dumps(final_results, indent=4))

if __name__ == "__main__":
    image_path = "bubble_sheet.jpg"
    main(image_path, debug=True)





