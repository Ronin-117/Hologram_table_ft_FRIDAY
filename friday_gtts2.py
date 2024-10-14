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

if __name__ == "__main__":
    
    while True:
        # Record the user's speech and transcribe it to text
        transcribed_text = record_and_transcribe()

        if transcribed_text and "friday" in transcribed_text.lower():
            # Remove the keyword "friday" from the query
            user_query = transcribed_text.lower().replace("friday", "").strip()
            if user_query=="hello":
                print("SUP DAWG")
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
