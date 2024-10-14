import speech_recognition as sr

# Initialize the recognizer
recognizer = sr.Recognizer()

# Use the microphone as the audio source
with sr.Microphone() as source:
    print("Please speak something...")
    
    # Adjust the recognizer sensitivity to ambient noise
    recognizer.adjust_for_ambient_noise(source, duration=1)
    
    # Listen for the first phrase and extract it into audio data
    audio_data = recognizer.listen(source)
    
    try:
        # Recognize speech using Google Web Speech API
        text = recognizer.recognize_google(audio_data)
        print("You said: " + text)
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
    except sr.RequestError:
        print("Sorry, there was an error with the request.")