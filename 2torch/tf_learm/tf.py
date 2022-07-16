import tensorflow as tf
tensor_a = tf.Variable([[1,2,3],[4,5,6],[7,8,9]])
tensor_b = tf.Variable([[1,0],[1,1],[2,2]],dtype=tf.int32)
tensor_c = tf.Variable([0],[1],dtype=tf.int32)
tensor_d = tf.Variable([[0,1]],dtype=tf.int32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.gather_nd(tensor_a,tensor_b)))
    print(sess.run(tf.gather_nd(tensor_a,tensor_c)))
    print(sess.run(tf.gather_nd(tensor_a,tensor_d)))
