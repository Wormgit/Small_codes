import tensorflow as tf

pre = tf.constant([[1.0,0.0,3.0],[1.0,5.0,3.0]])
ground = tf.constant([[4.0,2.0,3.0],[1.0,2.0,3.0]])
#cross_entropy = -tf.reduce_mean(ground * tf.log(pre))
pre2 = tf.clip_by_value(pre, 1, 4)
error = ground * tf.log(pre2)

with tf.Session() as sess:
    print (sess.run(pre2))   #limit the value
    print (sess.run(error))
