

import cv2
import mediapipe as mp
import pyautogui
import random
import util
from pynput.mouse import Button, Controller


mouse = Controller()

prev_index_finger_tip=None
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

    if prev_index_finger_tip is not None and index_finger_tip is not None:
        # Calculate the relative movement based on previous and current index finger positions
        dx = (index_finger_tip.x - prev_index_finger_tip.x) * screen_width
        dy = (index_finger_tip.y - prev_index_finger_tip.y) * screen_height
        
        # Move the mouse relatively by dx and dy
        pyautogui.moveRel(-dx, dy)
    
    # Update prev_index_finger_tip for the next frame
    prev_index_finger_tip = index_finger_tip



def is_left_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
            thumb_index_dist > 50
    )


def is_right_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90  and
            thumb_index_dist > 50
    )


def is_double_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            thumb_index_dist > 50
    )




def detect_gesture(frame, landmark_list, processed):
    if len(landmark_list) >= 21:

        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = util.get_distance([landmark_list[4], landmark_list[5]])

        
        move_mouse(index_finger_tip*0.02)#############################changed
        if util.get_distance([landmark_list[4], landmark_list[8]]) < 50 and util.get_angle(landmark_list[9], landmark_list[10], landmark_list[11]) < 90 and util.get_angle(landmark_list[17], landmark_list[18], landmark_list[19]) < 90:
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif util.get_distance([landmark_list[4], landmark_list[8]]) < 50 and util.get_angle(landmark_list[9], landmark_list[10], landmark_list[11]) > 90 and util.get_angle(landmark_list[17], landmark_list[18], landmark_list[19]) > 90:
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif util.get_distance([landmark_list[4], landmark_list[8]]) < 50 and util.get_angle(landmark_list[9], landmark_list[10], landmark_list[11]) < 90 and util.get_angle(landmark_list[17], landmark_list[18], landmark_list[19]) > 90:
            pyautogui.doubleClick()
            cv2.putText(frame, "double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        """elif is_left_click(landmark_list,  thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif is_right_click(landmark_list, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif is_double_click(landmark_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)"""

def main():
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(1)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)
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




