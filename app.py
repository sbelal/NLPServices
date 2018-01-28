"""
Flask service test
"""

import tensorflow as tf
from flask import Flask, jsonify, abort, make_response

app = Flask(__name__)

#region error handlers

@app.errorhandler(404)
def not_found(error):
	'''
	Error 404 handling
	'''
	return make_response(jsonify({'error': str(error)}), 404)

@app.errorhandler(500)
def internal_error(error):
	'''
	Error 500 handling
	'''
	return make_response(jsonify({'error': 'Internal server error. Details: '+str(error)}), 500)

#endregion

#region service functions

@app.route('/api/v1.0/tasks/<int:task_id>', methods=['GET'])
def get_tasks(task_id):
	"""
	qwqeqw eqweq we qwe qwe
	"""
	task = [task for task in tasks if task['id'] == task_id]
	if not task:
		abort(404)
	return jsonify({'task': task[0]})

@app.route('/api/v1.0/tensortest/<text>', methods=['GET'])
def home1(text):
	"""
	qwqeqw eqweq we qwe qwe
	"""
	tensorOutput = str(sess.run(hello))
	finalOutput = tensorOutput + text
	return finalOutput


@app.route('/api/v1.0/myerror', methods=['GET'])
def myerror():
	"""
	qwqeqw eqweq we qwe qwe
	"""

	tensorOutput = sess.run(hello)
	finalOutput = tensorOutput + "a"
	return finalOutput
#endregion

#region fake data
tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web',
        'done': False
    }
]
#endregion


hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()



if __name__ == '__main__':
    app.run(debug=False)
