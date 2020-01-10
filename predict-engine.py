from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras.models import load_model

import os
import matplotlib.pyplot as plt

import tensorflow as tf

#MODEL_NAME = args.model
MODEL_NAME = 'flower_model.h5'
# Array of class names, in this case three class names
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

# Entry point

model = load_model('models/' + MODEL_NAME)
#
predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5,],
    [5.9, 3.0, 4.2, 1.5,],
    [6.9, 3.1, 5.4, 2.1]
])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))

#