from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pyplot as plt

import tensorflow as tf

# Definitions
# pack_features_vector
def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels
# grad
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)
# loss
def loss(model, x, y):
    y_ = model(x)

    return loss_object(y_true=y, y_pred=y_)
#
############
# main entry
############
#
MODEL_NAME = 'models/segment_model_v3.h5'
#
print("Tensorflow version: {}".format(tf.version.VERSION))
#print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

train_dataset_url = "https://localhost:8443/segment_training_v32"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)
# column order in CSV file
#column_names = ['id', 'revenue', 'segment']
column_names = ['id', 'revenue', 'segment']

#feature_names = column_names[:-1]
feature_names = column_names[:-1]
print(feature_names)

label_name = column_names[-1]
#
print("Local copy of the dataset file: {}".format(train_dataset_fp))
#
print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))
#
# Array of class names, in this case three class names
class_names = ['Villakund', 'små och microföretag', 'Boende på gård', 'storkund']
# Create a tf.data.Dataset
#
#tf.random.shuffle(
#  train_dataset_fp,
#  seed=None,
#  name=None
#)
#
# Batch size should be configured based on size of input data
batch_size = 128
#
train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size=batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1,
    header=True)
#train_dataset = tf.data.experimental.make_csv_dataset(
#    train_dataset_fp,
#    batch_size,
#    shuffle=True,
#    shuffle_buffer_size=10000,
#    column_names=column_names,
#    label_name=label_name,
#    num_epochs=10,
#    header=True)    
#
# print batch and features
features, labels = next(iter(train_dataset))

print(features)
#
#plt.scatter(features['revenue'],
#            features['segment'],
#            c=labels,
#            cmap='viridis')

#plt.xlabel("revenue")
#plt.ylabel("segment")
#plt.show()
#
train_dataset = train_dataset.map(pack_features_vector)
features, labels = next(iter(train_dataset))
print(features[:10])
#
# Create a model using keras
model = tf.keras.Sequential([
  tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(1,)),  # input shape required
  tf.keras.layers.Dense(32, activation=tf.nn.relu),
  tf.keras.layers.Dropout(rate=0.2),
  tf.keras.layers.Dense(units=4, activation='softmax')
])
# compile model
print("Compile model: {}".format(MODEL_NAME))
# loss='binary_crossentropy',
model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy'])
# fit model
print("Fit model: {}".format(MODEL_NAME))
#
history = model.fit(
  train_dataset,
  epochs=10
)
#
history = history.history
print(history)
#print("Validation accuracy: {}, loss: {}".format(history["val_accuracy"][-1], history["val_loss"][-1]))
#
# Prediction
predictions = model(features)
predictions[:20]
print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))
# Define the loss and gradient function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
l = loss(model, features, labels)
print("Loss test: {}".format(l))
# Optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_value, grads = grad(model, features, labels)

print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                          loss(model, features, labels).numpy()))
#
# Training loop
## Note: Rerunning this cell uses the same model variables

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 51
#num_epochs = 101
#num_epochs = 401

for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Track progress
    epoch_loss_avg(loss_value)  # Add current batch loss
    # Compare predicted label to actual label
    epoch_accuracy(y, model(x))

  # End epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 5 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))
# out of training loop
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()

# Save model
model.save(MODEL_NAME)
#                                                        
print("End training segments...")
