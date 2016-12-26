from flask import Flask, request
import model
import cv2 as cv
import numpy as np

app = Flask(__name__)

server = model.DefaultModelServer()

@app.route('/')
def hello_world():
	return 'Fuck Yo!'


@app.route('/single-digit', methods=['POST'])
def single_digit():
	print('data length', len(request.data))
	q = cv.imdecode(np.frombuffer(request.data, dtype='uint8'), cv.IMREAD_COLOR)
	return str(server.serve(q))


if __name__ == '__main__':
	app.run()
