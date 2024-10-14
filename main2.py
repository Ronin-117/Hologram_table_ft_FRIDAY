import cv2
import mediapipe as mp
import pyautogui
import random
import util
from pynput.mouse import Button, Controller

mouse = Controller()

prev_index_finger_tip = None
screen_width, screen_height = pyautogui.size()

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP]
        return index_finger_tip
    return None, None

def move_mouse(index_finger_tip):
    global prev_index_finger_tip  # Use the global variable
    multiplrx = 4 
    multiplry = 5 # Multiplier for cursor movement
    smoothing_factor = 0.5  # Adjust this value for different smoothing levels

    if prev_index_finger_tip is not None and index_finger_tip is not None:
        # Calculate the relative movement based on previous and current index finger positions
        dx = (index_finger_tip.x - prev_index_finger_tip.x) * screen_width * multiplrx
        dy = (index_finger_tip.y - prev_index_finger_tip.y) * screen_height * multiplry

        # Apply smoothing
        new_x = pyautogui.position()[0] - dx * smoothing_factor
        new_y = pyautogui.position()[1] + dy * smoothing_factor  # Note: + because screen y coordinates increase downward

        # Clamp the position within screen bounds
        new_x = max(0, min(new_x, screen_width - 1))
        new_y = max(0, min(new_y, screen_height - 1))

        # Move the mouse to the new position
        pyautogui.moveTo(new_x, new_y)

    # Update prev_index_finger_tip for the next frame
    prev_index_finger_tip = index_finger_tip



def is_right_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90 and
            thumb_index_dist > 50
    )

def detect_gesture(frame, landmark_list, processed):
    if len(landmark_list) >= 21:
        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = util.get_distance([landmark_list[4], landmark_list[5]])
        
        move_mouse(index_finger_tip)  # Move the cursor based on finger position
        if util.get_distance([landmark_list[4], landmark_list[8]]) < 50 and util.get_angle(landmark_list[9], landmark_list[10], landmark_list[11]) > 90 and util.get_angle(landmark_list[17], landmark_list[18], landmark_list[19]) > 90:
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def main():
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use 0 for the default webcam
    # Set the resolution to 1920x1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FORMAT, -1)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))

            detect_gesture(frame, landmark_list, processed)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
