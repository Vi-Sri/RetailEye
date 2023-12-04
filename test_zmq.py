import time
import zmq
from flask import Flask
from flask import request 

app = Flask(__name__)
ctx = zmq.Context()
pub = ctx.socket(zmq.PUB)

def publish_message(message):
    url = "tcp://127.0.0.1:5555"
    try:
        pub.bind(url)
        time.sleep(1)
        print("sending message : {0}".format(message, pub))
        pub.send(message)
    except Exception as e:
        print("error {0}".format(e))
    finally:
        pub.unbind(url)

@app.route("/print", methods = ['GET'])
def printNumber():
    number = request.args.get('number')
    publish_message(f'Number {number} sent')
    return f"Number {number} sent"

if __name__ == "__main__":
    app.run(host='127.0.0.1', port='3000', debug=True) 