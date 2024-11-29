### Edit this one for more fuctionalities
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from pynput.mouse import Button, Controller
import time
import math


mouse = Controller()

# Cooldown time (in seconds) between clicks
CLICK_COOLDOWN = 1.0
last_click_time = 0  # Store the timestamp of the last click

prev_index_finger_tip = None
screen_width, screen_height = pyautogui.size()
center_x, center_y = screen_width // 2, screen_height // 2  # Center of the screen
margin = 10  # Margin so the cursor won't hit screen edges

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

def reset_mouse_position():
    """Reset the mouse to the center of the screen."""
    global prev_index_finger_tip
    pyautogui.moveTo(center_x, center_y)
    print("Mouse reset to the center")

def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        index_finger_tip = hand_landmarks.landmark[0]
        return index_finger_tip
    return None

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle

def move_mouse(index_finger_tip):
    """Move the mouse based on finger tip position."""
    global prev_index_finger_tip
    multiplrx, multiplry = 3, 4  # Multipliers for cursor movement
    smoothing_factor = 0.5  # Smoothing for the movement

    if prev_index_finger_tip and index_finger_tip:
        dx = (index_finger_tip.x - prev_index_finger_tip.x) * screen_width * multiplrx
        dy = (index_finger_tip.y - prev_index_finger_tip.y) * screen_height * multiplry

        new_x = pyautogui.position()[0] - dx * smoothing_factor
        new_y = pyautogui.position()[1] + dy * smoothing_factor

        new_x = max(margin, min(new_x, screen_width - margin))
        new_y = max(margin, min(new_y, screen_height - margin))

        try:
            pyautogui.moveTo(new_x, new_y)
        except Exception as e:
            print(f"Error moving the mouse: {e}")
            reset_mouse_position()

    prev_index_finger_tip = index_finger_tip

def detect_gesture(frame, landmark_list, processed):
    global last_click_time

    if len(landmark_list) >= 21:
        index_finger_tip = find_finger_tip(processed)

        # Calculate angles for index, middle, and ring fingers
        angle_index = calculate_angle(landmark_list[5], landmark_list[6], landmark_list[7])
        angle_middle = calculate_angle(landmark_list[9], landmark_list[10], landmark_list[11])
        angle_ring = calculate_angle(landmark_list[13], landmark_list[14], landmark_list[15])

        #print(f"{angle_index =},{angle_middle =},{angle_ring =}")
        
        # Check if all angles are below 120 degrees
        if angle_index < 150 and angle_middle < 120 and angle_ring < 120 :
            # Get the current time
            current_time = time.time()

            # Perform the click only if the cooldown period has passed
            if current_time - last_click_time >= CLICK_COOLDOWN:
                mouse.press(Button.left)
                mouse.release(Button.left)
                last_click_time = current_time  # Update the last click time
                cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif angle_index > 150 and angle_middle < 120 and angle_ring < 120 :
            current_time = time.time()

            # Perform the click only if the cooldown period has passed
            if current_time - last_click_time >= CLICK_COOLDOWN:
                mouse.press(Button.right)
                mouse.release(Button.right)
                last_click_time = current_time  # Update the last click time
                cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        move_mouse(index_finger_tip)

def main():
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use 0 for default webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            #frame = cv2.flip(frame, 1)  # Mirror view
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x * frame.shape[1], lm.y * frame.shape[0]))

            detect_gesture(frame, landmark_list, processed)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
