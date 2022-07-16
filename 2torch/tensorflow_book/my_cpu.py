import os

print(os.getcwd())
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


import platform
print(platform.python_version())