from flask import Flask, render_template
from flask_socketio import SocketIO
import zmq

app = Flask(__name__)
socketio = SocketIO(app)

context = zmq.Context()
zmq_socket = context.socket(zmq.PUB)
zmq_socket.bind("tcp://*:5555")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('send_prediction')
def handle_prediction(json):
    print('Received prediction:', json)
    zmq_socket.send_json(json)

if __name__ == '__main__':
    socketio.run(app, debug=True)