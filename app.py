from flask import Flask, render_template, request, jsonify
import requests
import cv2
import mediapipe as mp
import pyautogui
import random
import util
from pynput.mouse import Button, Controller
import HG_util

app = Flask(__name__)

# Placeholder function for AI responses (this can be replaced with your AI logic)
def get_ai_response(command):
    if "weather" in command:
        return "It's sunny with 24°C."
    elif "news" in command:
        return "Here are the latest news headlines."
    elif "play music" in command:
        return "Playing your favorite music."
    else:
        return "Sorry, I didn't understand the command."

# Route for the main page
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


if __name__ == '__main__':
    app.run(debug=True)
