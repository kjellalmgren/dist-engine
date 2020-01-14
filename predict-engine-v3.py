from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras.models import load_model

import os
import matplotlib.pyplot as plt

import tensorflow as tf

#MODEL_NAME = args.model
MODEL_NAME = 'segment_model_v3.h5'
# Array of class names, in this case three class names
class_names = ['Villakund', 'små och microföretag', 'Boende på gård', 'storkund']

# Entry point
print("loading model: {}".format(MODEL_NAME))
model = load_model('models/' + MODEL_NAME)
#
# 0 = Villa, revenue = 0
# 1 = Boende på gård
# 2 = Små och microföretag
# 3 = Storkund
#
predict_dataset = tf.convert_to_tensor([
    [400000.],
    [22000000.],
    [8500000],
    [250000]
])

predictions = model(predict_dataset)
#print(predictions)
for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("Example {} prediction: segment {} - {} ({:4.1f}%)".format(i, class_idx, name, 100*p))
#