import tensorflow as tf

#t2 = tf.constant([[[1, 2, 3]]])
t2=tf.constant([[[1,2,3], [2,3,4],[2,1,4]],
               [[1,2,3], [2,3,4],[2,1,4]],
               [[1,2,3],  [2,3,4],[2,1,4]]])


a = tf.pad(t2, [[0, 0], [0, 0],[0, 1]])

with tf.Session() as sess:
    a = sess.run(a)
    print(t2.get_shape())
    print(a)
    print(a.shape)


#oo=regression.get_shape()
a = [3,66,56]
kk=a[:-1]
print (kk)
