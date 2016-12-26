from flask import Flask

app = Flask(__name__)


def read_model():
	pass


@app.route('/')
def hello_world():
	return 'Fuck Yo!'


if __name__ == '__main__':
	app.run()
