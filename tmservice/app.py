from flask import Flask, render_template
from flask import request
from flask_socketio import SocketIO
import zmq
from threading import Thread
import time
import subprocess

app = Flask(__name__)

prediction_data = None

def send_prediction():
    global prediction_data
    sock = zmq.Context().socket(zmq.PAIR)
    sock.bind("tcp://127.0.0.1:10000")
    while True:
        if prediction_data is not None:
            max_prediction = max(prediction_data['prediction'], key=lambda x: x['probability'])
            print("Sending prediction:", max_prediction)
            # sock.send_json(prediction_data)
            prediction_data = None
        time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/receive_state', methods = ['POST'])
def receive_state():
    global prediction_data
    prediction_data = request.get_json()
    # print(prediction_data)
    return 'OK'

if __name__ == '__main__':
    person_state_sender = Thread(target=send_prediction)
    person_state_sender.start()
    app.run(host='127.0.0.1', port='3000', debug=True)
    person_state_sender.join()