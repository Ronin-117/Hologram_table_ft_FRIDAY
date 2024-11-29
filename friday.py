from flask import Flask, render_template, request, jsonify,redirect, url_for
import threading
import cv2
import util
import pathlib
import textwrap
import pyttsx3
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
import webbrowser
import pywhatkit
import requests
from threading import Lock
import time  # Import time to introduce delay
import traceback
import pyautogui


click_lock = Lock() 

gesture_lock = Lock()  # Create a global lock



engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# Initialize global stop flag
interrupt_flag = False

keyboarde = Controller()

prompt="(Note when generating output. 1- give a single paragraph about the topic asked that does not exceed 50 words, 2- add fullstop or dot aftre every sentences including the ones in numbers or unnumbered list )"

site_choice = None

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


def is_connected():
    try:
        # Make a request to a reliable website
        response = requests.get("http://www.google.com", timeout=5)
        return response.status_code == 200
    except requests.ConnectionError:
        return False

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
GOOGLE_API_KEY = '<<<<<<<YOUR API KEY>>>>>>>>'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Function to process the query and get a response from the AI model
def respond_to_friday(query):
    global interrupt_flag
    buffer = ""  # Buffer to hold the response
    retries = 3 

    for attempt in range(retries):
        try:
            # Get AI-generated response based on the user's query in streaming mode
            response = model.generate_content(query, stream=True)

            print("Response from Friday:")
            for chunk in response:
                if interrupt_flag:
                    print("\n--- Interrupt received, stopping response ---\n")
                    interrupt_flag = False
                    return#break
                
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
                        buffer = buffer[end_idx + 1:].strip()
                        # Keep the remainder in the buffer
                    else:
                        break  # No complete sentences found

            # Speak any remaining text in the buffer after the loop
            if buffer:
                print(buffer, end="", flush=True)
                speak_gtts(buffer)
            # Simulate pressing 'q'
            keyboarde.press('q')
            keyboarde.release('q')
        except google.api_core.exceptions.InternalServerError as e:
            print(f"Error occurred: {e}. Retrying in 2 seconds...")
            time.sleep(2)  # Wait before retrying

    print("Failed to generate response after multiple attempts.")

# Function to listen for the 'Q' key to stop
def listen_for_stop():
    global interrupt_flag
    if interrupt_flag:
        return 
    print("Press 'Q' to stop friday...")
    keyboard.wait('q')  # Waits for 'Q' key press
    interrupt_flag = True
    print("\n--- 'Q' key pressed. Stopping friday ---\n")

app = Flask(__name__)

# Flask route for main page
@app.route('/')
def index():
    global site_choice
    if site_choice == 'hm':
        return redirect(url_for('hm'))
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

@app.route('/sa')
def sa():
    return render_template('test.html')

@app.route('/get_site_choice')
def get_site_choice():
    global site_choice
    if site_choice:
        redirect_url = url_for(site_choice)
        site_choice = None  # Reset after redirect
        return jsonify({'redirect': redirect_url})
    return jsonify({'redirect': None})




time_q =["what is the time","what is the current time"]
holo_q=["open holomat","open holo mat","open hollow mat","open hollowmat","open hollowmap","open hollow map","open follow mat","open follow map","open format","open hola mat","open holo math","open hollow math","open follow math","open holo match","open hollow match","open follow match","open hollow man","open howrah match"]
main_q =["go back","go back to main page","go back to home page","go back to homepage","go back to mainpage"]
yt_q=["open you tube","open youtube"]
sp_q=["open spotify"]
favsng_q=["play everyone's favorate song","play the greatest song of all time","play the greatest song","play everyone's favorate music","play the greatest music"]
home_q=["please go back to home page","please go back to homepage","please go back to main page","please go back to mainpage","go back to initial page","go to initial page"]
gen_q=["play the music","play the song","play the video"]
def friday_t():
    global site_choice
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
            elif user_query in holo_q:
                speak_gtts("Opening Holomat")
                site_choice="hm"
                continue
            elif user_query in main_q:
                speak_gtts("redirecting to home page")
                site_choice="index"
                continue
            elif user_query in yt_q:
                speak_gtts("Opening Youtube")
                webbrowser.open("https://www.youtube.com")
                continue
            elif user_query in sp_q:
                speak_gtts("Opening Spotify")
                webbrowser.open("https://open.spotify.com")
                continue
            elif user_query in favsng_q:
                webbrowser.open("https://shattereddisk.github.io/rickroll/rickroll.mp4")
                time.sleep(1)
                pyautogui.press('space')
                speak_gtts("HAHA get rolled")
                continue
            elif user_query in home_q:
                speak_gtts("Opening home page")
                webbrowser.open("http://127.0.0.1:5000")
                continue
            elif "play the music" in user_query or "play the song" in user_query or "play the video" in user_query:
                user_query.replace("play the song","")
                user_query.replace("play the music","")
                user_query.replace("play the video","")
                pywhatkit.playonyt(user_query)
                continue
            elif "assemble suit" in user_query or "open mark 42 project" in user_query or "assembly suit" in user_query:
                if "assemble suit" in user_query or "assembly suit" in user_query:
                    speak_gtts("Assembeling suit")
                else:
                    speak_gtts("Opening mark 42 project")
                webbrowser.open("http://127.0.0.1:5000/sa")
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


        else:
            print("Waiting for the keyword 'friday' to activate...")


# Main function for gesture recognition

# Start Flask and gesture recognition in separate threads
if __name__ == '__main__':
    threading.Thread(target=app.run, kwargs={'debug': True, 'use_reloader': False}).start()  # Start Flask in a thread
    threading.Thread(target=friday_t).start()  