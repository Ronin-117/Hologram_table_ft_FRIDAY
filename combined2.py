from flask import Flask, render_template, request, jsonify
import threading
import cv2
import mediapipe as mp
import pyautogui
from pynput.mouse import Button, Controller
import util
import pathlib
import textwrap
import google.generativeai as genai
import speech_recognition as sr
import threading
import keyboard 
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
from pynput.keyboard import Controller
from datetime import datetime



# Initialize global stop flag
interrupt_flag = False

keyboarde = Controller()

prompt="(Note when generating output. 1- give a single paragraph about the topic asked that does not exceed 50 words, 2- add fullstop or dot aftre every sentences including the ones in numbers or unnumbered list )"

def speak_gtts(text):
    # Generate the speech using gTTS
    tts = gTTS(text=text, lang='en', tld='ca')
    
    # Store the audio in a BytesIO buffer instead of saving to disk
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    
    # Move the buffer's cursor to the beginning
    audio_buffer.seek(0)
    
    # Load the audio with pydub
    audio = AudioSegment.from_file(audio_buffer, format="mp3")
    
    # Play the audio directly
    play(audio)

def record_and_transcribe():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Record audio from microphone
    with sr.Microphone() as source:
        print("Please start speaking... (Say 'friday' to activate)")
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.listen(source)

    # Convert speech to text
    try:
        print("Transcribing speech to text...")
        text = recognizer.recognize_google(audio_data)
        print("Transcription:", text)

        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print("Error with Google Speech Recognition service; {0}".format(e))
        return None

# Configure Google API key for generative model
GOOGLE_API_KEY = 'AIzaSyBZknRAIpjSxW0VvGqQSuBLWOWWONJp1Dg'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Function to process the query and get a response from the AI model
def respond_to_friday(query):
    global interrupt_flag
    buffer = ""  # Buffer to hold the response

    # Get AI-generated response based on the user's query in streaming mode
    response = model.generate_content(query, stream=True)

    print("Response from Friday:")
    for chunk in response:
        if interrupt_flag:
            print("\n--- Interrupt received, stopping response ---\n")
            interrupt_flag = False
            break
        
        # Clean the chunk and append to the buffer
        chunk_text = chunk.text.replace("*", "")
        buffer += chunk_text
        
        max_chunk_size = 200

        # Check for sentence-ending punctuation
        while True:
            # Find the index of the last sentence-ending punctuation
            end_idx = max(buffer.rfind('.'), buffer.rfind('!'), buffer.rfind('?'))
            if end_idx != -1:  # If there is a sentence-ending punctuation
                # Split the buffer into a complete sentence and the remainder
                complete_sentence = buffer[:end_idx + 1]  # Include the punctuation
                print(complete_sentence, end="", flush=True)
                speak_gtts(complete_sentence)  # Speak the complete sentence
                buffer = buffer[end_idx + 1:].strip()  # Keep the remainder in the buffer
            else:
                break  # No complete sentences found

    # Speak any remaining text in the buffer after the loop
    if buffer:
        print(buffer, end="", flush=True)
        speak_gtts(buffer)
    # Simulate pressing 'q'
    keyboarde.press('q')
    keyboarde.release('q')

# Function to listen for the 'Q' key to stop
def listen_for_stop():
    global interrupt_flag
    print("Press 'Q' to stop friday...")
    keyboard.wait('q')  # Waits for 'Q' key press
    interrupt_flag = True
    print("\n--- 'Q' key pressed. Stopping friday ---\n")

app = Flask(__name__)
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

# Placeholder function for AI responses
def get_ai_response(command):
    if "weather" in command:
        return "It's sunny with 24°C."
    elif "news" in command:
        return "Here are the latest news headlines."
    elif "play music" in command:
        return "Playing your favorite music."
    else:
        return "Sorry, I didn't understand the command."

# Flask route for main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle commands
@app.route('/send_command', methods=['POST'])
def send_command():
    user_input = request.form['command']
    response = get_ai_response(user_input)
    return jsonify({'response': response})

@app.route('/hm')
def hm():
    return render_template('hm.html')

# Function to find finger tip
def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP]
        return index_finger_tip
    return None

# Function to move mouse
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
time_q =["what is the time","what is the current time"]

def friday_t():
    while True:
        # Record the user's speech and transcribe it to text
        transcribed_text = record_and_transcribe()

        if transcribed_text and "friday" in transcribed_text.lower():
            # Remove the keyword "friday" from the query
            user_query = transcribed_text.lower().replace("friday", "").strip()
            if user_query=="hello":
                speak_gtts("Hello , how may i help you")
                continue
            elif user_query in time_q:
                # Get the current date and time
                now = datetime.now()
                # Extract the current time
                current_time = now.strftime("%H:%M")
                speak_gtts("Current time is,"+str(current_time))
                continue
            user_query=user_query+prompt

            if user_query:
                interrupt_flag = False
                # Start a new thread to listen for the 'Q' key press
                interrupt_thread = threading.Thread(target=listen_for_stop)
                interrupt_thread.start()

                # Get the response from the AI model and speak it
                respond_to_friday(user_query)

                # Wait for the interrupt thread to finish
                interrupt_thread.join()

                interrupt_flag = False

        else:
            print("Waiting for the keyword 'friday' to activate...")


# Main function for gesture recognition
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

# Start Flask and gesture recognition in separate threads
if __name__ == '__main__':
    threading.Thread(target=app.run, kwargs={'debug': True, 'use_reloader': False}).start()  # Start Flask in a thread
    threading.Thread(target=friday_t).start()  # Start Flask in a thread
    main()  # Start gesture recognition
