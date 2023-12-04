import zmq

sock = zmq.Context().socket(zmq.REP)
sock.connect("tcp://127.0.0.1:10000")

while True:
    message = sock.recv_json()
    print("Received request: %s" % message)
    sock.send_json({'result': 'success'})
