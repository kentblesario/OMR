import cv2
import numpy as np
import json
import math
import os
import time
import threading
from datetime import datetime, timedelta, timezone
import sys
from Crypto.Cipher import AES 

start_timestamp = 1741583891657  # March 8, 2025, 13:55:12 UTC
start_time = datetime.fromtimestamp(start_timestamp / 1000, timezone.utc)

# Expiration time (120 hours later)
expiration_time = start_time + timedelta(hours=120)

# ------------------------------
# Image processing functions
# ------------------------------
def detect_circles(image, debug=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug:
        cv2.imshow("Gray Image", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    blurred = cv2.GaussianBlur(gray, (9, 9), 1)
    if debug:
        cv2.imshow("Blurred Image", blurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.5, minDist=20,
                               param1=50, param2=30, minRadius=25, maxRadius=28)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        if debug:
            debug_image = image.copy()
            for (x, y, r) in circles:
                cv2.circle(debug_image, (x, y), r, (0, 255, 0), 2)
            cv2.imshow("Detected Circles", debug_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return circles
    return []

def group_circles(circles, y_threshold=15, debug=True):
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

def group_circles_with_item(circles, answer_image, y_threshold=15):
    debug = True
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

    if debug:
        debug_image = answer_image.copy()
        for group_obj in result:
            item_number = group_obj["item"]
            group = group_obj["group"]
            x_min = min(circle[0] - circle[2] for circle in group)
            y_min = min(circle[1] - circle[2] for circle in group)
            x_max = max(circle[0] + circle[2] for circle in group)
            y_max = max(circle[1] + circle[2] for circle in group)
            cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(debug_image, f"Item {item_number}", (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Grouped Circles", debug_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return result

def process_row(image, circles):
    mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    percentages = {}
    for idx, (x, y, r) in enumerate(circles):
        cropped = image[y - r:y + r, x - r:x + r]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        pixel_count = np.sum(binary == 255)
        circle_area = math.pi * (r ** 2)
        percentage = (pixel_count / circle_area) * 100 if circle_area > 0 else 0
        percentages[mapping.get(idx, idx)] = percentage

    qualifiers = {key: perc for key, perc in percentages.items() if perc >= 85}
    if len(qualifiers) == 1:
        selected = list(qualifiers.keys())[0]
        for idx, (x, y, r) in enumerate(circles):
            cv2.circle(image, (x, y), r, (0, 0, 255), 2)
            cv2.putText(image, f"{mapping.get(idx, idx)}: {percentages[mapping.get(idx, idx)]:.1f}%", (x - 10, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        height, width = image.shape[:2]
        resized = cv2.resize(image, (width // 2, height // 2))
        x_min = min(circle[0] - circle[2] for circle in circles)
        y_min = min(circle[1] - circle[2] for circle in circles)
        cv2.putText(resized, f"Selected: {selected}", (x_min + 20, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(resized, f"Circle {idx+1}", (width // 2 - 20, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Process Row", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return selected
    else:
        return None

# ------------------------------
# Camera helper functions
# ------------------------------
def list_available_cameras(max_indices=5):
    available = []
    for i in range(max_indices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

def capture_image_from_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_index}")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Failed to capture image.")
        return None
    return frame

# ------------------------------
# Evaluation function
# ------------------------------
def evaluate_exams(test_folder):
    # Process answer sheet image
    answer_sheet_folder = os.path.join(test_folder, "answer_sheet")
    exam_folder = os.path.join(test_folder, "test_exams")
    
    answer_images = [f for f in os.listdir(answer_sheet_folder) if f.lower().endswith(('.jpg', '.png'))]
    if not answer_images:
        print("No answer sheet images found.")
        return
    answer_image_path = os.path.join(answer_sheet_folder, answer_images[0])
    answer_img = cv2.imread(answer_image_path)
    if answer_img is None:
        print("Error reading answer sheet image.")
        return
    # MAIN FLOW
    circles = detect_circles(answer_img)
    groups = group_circles_with_item(circles, answer_img,y_threshold=15)
    answer_results = {}
    for group_obj in groups:
        letter = process_row(answer_img, group_obj["group"])
        answer_results[group_obj["item"]] = letter if letter is not None else None

    # Optionally save answer sheet processing results
    answer_json_path = os.path.join(answer_sheet_folder, "results.json")
    with open(answer_json_path, "w") as fp:
        json.dump(answer_results, fp, indent=4)
    print(f"Answer sheet results saved to {answer_json_path}")

    total_questions = len(answer_results)
    evaluation_results = {}
    
    exam_images = [f for f in os.listdir(exam_folder) if f.lower().endswith(('.jpg', '.png'))]
    if not exam_images:
        print("No exam images found in test_exams folder.")
        return

    for exam_file in exam_images:
        exam_image_path = os.path.join(exam_folder, exam_file)
        exam_img = cv2.imread(exam_image_path)
        if exam_img is None:
            print(f"Error reading exam image {exam_file}. Skipping.")
            continue

        circles_exam = detect_circles(exam_img)
        groups_exam = group_circles_with_item(circles_exam, exam_img,y_threshold=15)
        exam_results = {}
        for group_obj in groups_exam:
            letter = process_row(exam_img, group_obj["group"])
            exam_results[group_obj["item"]] = letter if letter is not None else None

        # Save exam processing results in test_exams folder
        exam_json_path = os.path.join(exam_folder, exam_file.split('.')[0] + ".json")
        with open(exam_json_path, "w") as fp:
            json.dump(exam_results, fp, indent=4)
        print(f"Exam processing results saved to {exam_json_path}")
        print(json.dumps(exam_results, indent=4))

        # Compare exam results with answer sheet to compute score
        correct = 0
        for key, correct_answer in answer_results.items():
            if key in exam_results and exam_results[key] == correct_answer:
                correct += 1
        evaluation_results[exam_file] = {"score": correct, "total": total_questions}

    # Save final evaluation results as final_results.json in test_folder
    final_results_file = os.path.join(test_folder, "final_results.json")
    with open(final_results_file, "w") as fp:
        json.dump(evaluation_results, fp, indent=4)
    print(f"Final evaluation results saved to {final_results_file}")



# ------------------------------
# Test Checker Flow
# ------------------------------
def initiate_test_checker(debug=True):
    base_folder = "test_folders"
    os.makedirs(base_folder, exist_ok=True)
    
    # Ask for a test id and ensure it doesn't already exist
    while True:
        test_id = input("Enter test id: ").strip()
        test_folder = os.path.join(base_folder, test_id)
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
            break
        else:
            print("Test id folder already exists. Please input another test id.")
    
    # Create answer_sheet folder
    answer_sheet_folder = os.path.join(test_folder, "answer_sheet")
    os.makedirs(answer_sheet_folder, exist_ok=True)
    print(f"Answer sheet folder created: {answer_sheet_folder}")
    
    # List available cameras and select one
    available_cams = list_available_cameras(max_indices=5)
    if not available_cams:
        print("No available cameras found.")
        return
    print("Available Cameras:")
    for idx, cam in enumerate(available_cams):
        print(f"{idx}: Camera index {cam}")
    try:
        selection = int(input("Select camera number: ").strip())
        cam_index = available_cams[selection]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return
    
    # Capture answer sheet image and save as image1.jpg
    frame = capture_image_from_camera(cam_index)
    if frame is None:
        return
    answer_image_path = os.path.join(answer_sheet_folder, "image1.jpg")
    cv2.imwrite(answer_image_path, frame)
    print(f"Captured answer sheet image saved to {answer_image_path}")
    
    # Process the captured answer sheet image
    circles = detect_circles(frame)
    grouped_circles = group_circles_with_item(circles, frame,y_threshold=15)
    final_results = {}
    for group_obj in grouped_circles:
        letter = process_row(frame, group_obj["group"])
        final_results[group_obj["item"]] = letter if letter is not None else None
    
    results_json_path = os.path.join(answer_sheet_folder, "results.json")
    with open(results_json_path, "w") as fp:
        json.dump(final_results, fp, indent=4)
    print(f"Answer sheet processing results saved to {results_json_path}")
    print(json.dumps(final_results, indent=4))
    
    input("Set up the test exams and press Enter to continue...")

    # Create test_exams folder and start exam image capture loop
    exam_folder = os.path.join(test_folder, "test_exams")
    os.makedirs(exam_folder, exist_ok=True)
    exam_id = 1
    while True:
        exam_image_path = os.path.join(exam_folder, f"{exam_id}.jpg")
        frame_exam = capture_image_from_camera(cam_index)
        if frame_exam is None:
            print("Failed to capture exam image. Exiting exam capture.")
            break
        cv2.imwrite(exam_image_path, frame_exam)
        print(f"Captured exam image saved to {exam_image_path}")
        
        # Process the exam image
        circles_exam = detect_circles(frame_exam)
        grouped_exam = group_circles_with_item(circles_exam,frame_exam, y_threshold=15)
        exam_results = {}
        for group_obj in grouped_exam:
            letter = process_row(frame_exam, group_obj["group"])
            exam_results[group_obj["item"]] = letter if letter is not None else None
        
        exam_json_path = os.path.join(exam_folder, f"{exam_id}.json")
        with open(exam_json_path, "w") as fp:
            json.dump(exam_results, fp, indent=4)
        print(f"Exam processing results saved to {exam_json_path}")
        print(json.dumps(exam_results, indent=4))
        
        cont = input("Do you want to capture another test exam image? (y/n): ").strip().lower()
        if cont != 'y':
            break
        exam_id += 1

    # Evaluate each exam against the answer sheet before exiting
    evaluate_exams(test_folder)

# ------------------------------
# Recheck Test Flow
# ------------------------------
def recheck_test():
    base_folder = "test_folders"
    test_id = input("Enter test id to recheck: ").strip()
    test_folder = os.path.join(base_folder, test_id)
    if not os.path.exists(test_folder):
        print("Test id folder not found.")
        return
    print(f"Rechecking test id: {test_id}")
    evaluate_exams(test_folder)
    
    
# ------------------------------
# Function to Display Remaining Time
# ------------------------------
def display_remaining_time(end_time):
    while True:
        remaining_time = end_time - datetime.now(timezone.utc)
        if remaining_time.total_seconds() <= 0:
            print("\nTime is up! Exiting program...")
            os._exit(0)  # Force exit the entire program
        hours_left = remaining_time.total_seconds() / 3600
        print(f"\rTime remaining: {hours_left:.2f} hours", end="", flush=True)
        time.sleep(60)  # Update every 60 seconds

# ------------------------------
# Main Menu with Timer
# ------------------------------
def main_menu():
    # Start timer in a separate daemon thread
    # timer_thread = threading.Thread(target=display_remaining_time, args=(expiration_time,), daemon=True)
    # timer_thread.start()

    while True:
        print("\nMain Menu:")
        print("1. Initiate test checker")
        print("2. Recheck test")
        print("0. Exit")
        choice = input("Select an option: ").strip()
        if choice == '1':
            initiate_test_checker(debug=True)
        elif choice == '2':
            recheck_test()
        elif choice == '0':
            print("Exiting program.")
            os._exit(0)
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main_menu()