import tensorflow as tf
#a = tf.constant([1,2.0],name="a",dtype=tf.float32)
a = tf.constant([1,2.0],name="a")
b = tf.constant([3,4.0],name="b")

result = a + b
r = b+a

with tf.Session() as sess:
    print(sess.run (result))
    print(r.graph is tf.compat.v1.get_default_graph())
    print(r,result)

    with sess.as_default():
        print(result.eval)
'''''''''
# choose GPU device
g = tf.Graph()
with g.device('/gpu:0'):
    result = a + b
'''