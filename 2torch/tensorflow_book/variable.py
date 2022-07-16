import tensorflow as tf

w1 = tf.Variable(tf.random.normal([2,3], mean=1, stddev = 2))
w2 = tf.Variable(tf.random.normal((3,1), mean=1, stddev = 2))

x = tf.placeholder(dtype=tf.float32,shape=[3,2],name='input')

#x = tf.constant([[0.7,0.9]])

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

biases = tf.Variable(tf.zeros([3]))

with tf.Session() as sess:
    #sess.run(w1.initializer)
    #sess.run(w2.initializer)
#use the sentences to initialize instead of doing seperately
    init = tf.global_variables_initializer()
    sess.run(init)

    print(sess.run(y,feed_dict={x:[[0.1,0.2],[0.5,0.9],[0.8,0.2]]})) # a batch of 3
    #print (sess.run(w2))
    #print(sess.run(w1))

    print(tf.compat.v1.global_variables()) #get all the variables in current gragh
    print(tf.compat.v1.trainable_variables()) # epoch is a hyperparameter rather than a trainable parameter


#tf.assign(w1,w2,validate_shape=False)   give value to another without considering the shape  not common