#!/usr/bin/env python

import argparse
import os,glob
import sys
import warnings
import keras
import keras.preprocessing.image
import tensorflow as tf
print(sys.path)

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    __package__ = "keras_retinanet.bin"



print(sys.path)
print(__file__)
print(__package__)