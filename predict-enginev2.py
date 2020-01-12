from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras.models import load_model

import os
import matplotlib.pyplot as plt

import tensorflow as tf

#MODEL_NAME = args.model
MODEL_NAME = 'segment_model.h5'
# Array of class names, in this case three class names
class_names = ['micro customer', 'small customer', 'medium customer', 'huge customer']

# Entry point
print("loading model: {}".format(MODEL_NAME))
model = load_model('models/' + MODEL_NAME)
#
# 0 = 0-10 Micro
# 1 = 11-20 Small
# 2 = 21-30 Medium
# 3 = 31-40 Huge
#
predict_dataset = tf.convert_to_tensor([
    [33.0],
    [11.66],
    [22.0],
    [0.4]
])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("Example {} prediction: segment {} - {} ({:4.1f}%)".format(i, class_idx, name, 100*p))

#