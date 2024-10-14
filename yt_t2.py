import webbrowser
import pywhatkit
import socket


# Get the hostname of the local machine
hostname = socket.gethostname()

# Get the IP address associated with the hostname
localhost_ip = socket.gethostbyname(hostname)


#web  = "https://www.youtube.com/results?search_query="+"iron man"
web=f"http://127.0.0.1:5000"
webbrowser.open(web)
#pywhatkit.playonyt("Never gonna give you up")