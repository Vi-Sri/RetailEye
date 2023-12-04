from flask import Flask, render_template
from flask_socketio import SocketIO
import zmq
from multiprocessing import Process, Queue
import time
import subprocess

app = Flask(__name__)
socketio = SocketIO(app)

sock = zmq.Context().socket(zmq.REQ)
sock.connect("tcp://127.0.0.1:10000")

prediction_data = None

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('send_prediction')
def handle_prediction(json):
    global prediction_data
    prediction_data = json
    sock.send_json(json)
    print('Received prediction:', json)

if __name__ == '__main__':
    socketio.run(app, host="127.0.0.1", port=9000, debug=True)