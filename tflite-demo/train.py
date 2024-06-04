import tensorflow.keras as keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from clearml import Task, Logger
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

KERAS_OUTPUT_PATH = "model.keras"

task = Task.init(project_name='vives-mlops-workshop', task_name='edge-deployment', output_uri=True)

# hyperparameters
config = {
    "conv1": 32,
    "conv2": 46,
    "conv3": 46,
    "dense1": 100,
}
task.connect(config)

# ---

# download the dataset
(images, labels), _ = keras.datasets.cifar10.load_data()

# there are 10 classes of images
all_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# choose four classes (feel free to change this!)
class_names = ["bird", "cat", "deer", "dog"]
print("Class names:", class_names)

# only keep images of these classes
class_indexes = [all_classes.index(c) for c in class_names]
to_keep = np.array([l in class_indexes for l in labels])
images = images[to_keep]
labels = labels[to_keep]

# change indexes from 10 to 2 classes
labels = np.array([class_indexes.index(l) for l in labels])

# normalize pixels between 0 and 1
images = images / 255.0

# split into train and test set
split = round(len(images) * 0.8)
train_images = images[:split]
train_labels = labels[:split]
test_images = images[split:]
test_labels = labels[split:]
print("Number of train images:", len(train_images))
print("Number of test images:", len(test_images))

# ---

# create neural network
model = keras.models.Sequential()
model.add(keras.Input(shape=(32, 32, 3)))

# convolutional layers
model.add(layers.Conv2D(config["conv1"], (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(config["conv2"], (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(config["conv3"], (3, 3), activation="relu"))

# add dense layers
model.add(layers.Flatten())
model.add(layers.Dense(config["dense1"], activation="relu"))
model.add(layers.Dense(4, activation="softmax"))

# ---

# compile and train the model
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# ---

# report accuracy to ClearML
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
logger = Logger.current_logger()

for i in range(len(accuracy)):
    logger.report_scalar("training", "accuracy", accuracy[i], iteration=i)
    logger.report_scalar("training", "val_accuracy", val_accuracy[i], iteration=i)

# save model to disk
model.save(KERAS_OUTPUT_PATH )

# show confusion matrix
test_predict = model.predict(test_images)
test_predict = np.argmax(test_predict, axis=1)
cm = confusion_matrix(test_labels, test_predict, normalize='pred')
ConfusionMatrixDisplay(cm, display_labels=class_names).plot()
plt.title("Confusion Matrix")
plt.show()

# report bad images
reported = 0
for i in range(len(test_labels)):
    if test_labels[i] != test_predict[i]:
        actual_class = class_names[test_labels[i]]
        predict_class = class_names[test_predict[i]]
        description = actual_class + " => " + predict_class
        logger.report_image("Bad images", description, iteration=reported, image=test_images[i])
        reported += 1
        if reported >= 10:
            break

# close ClearML task
task.close()