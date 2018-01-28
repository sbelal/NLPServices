"""
Flask service test
"""

import tensorflow as tf
from flask import Flask

app = Flask(__name__)
@app.route("/")


def home0():
	"""
	qwqeqw eqweq we qwe qwe
	"""
	return str(42)


def home1():
	"""
	qwqeqw eqweq we qwe qwe
	"""
	return sess.run(hello)




hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()



if __name__ == '__main__':
    app.run(debug=True)
