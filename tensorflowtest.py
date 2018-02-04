"""
Testing stuff goes here
"""

import numpy as np
import tensorflow as tf


print("Start..")

tf.reset_default_graph()
graph = tf.Graph()
sess = tf.Session(graph=graph)

with graph.as_default():
    v1 = tf.Variable(22)
    v2 = tf.Variable(5)
    v3 = tf.add(v1,v2)
    v4 = v1.assign_add(100)
    v5 = v1 + 5
with sess:
    sess.run(tf.global_variables_initializer())
    output = sess.run(v3)        
    output = sess.run(v3, {v1:output})
    output = sess.run(v4)
    output = sess.run(v4)
    output = sess.run(v4)
    output = sess.run(v1)        

    output2 = sess.run(v5)
print(output)



#sess.close()

print("..End")