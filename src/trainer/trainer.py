from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os


# Initialize the initial learning rate, number of times to train and batch size
starting_learning = 1e-4
training_size = 25
batch_size = 50

# Indicates the path with the list of images (dataset) and generates a list of the images
image_sizeX = 224
image_sizeY = 224
images_path = "dataset/mask"
print("[INFO] Loading images from " + images_path)
images_list = list(paths.list_images(images_path))
data = []
labels = []

# Loop that scans all images paths
for images_path in images_list:
	# Extracts the class label from the file name
	label = images_path.split(os.path.sep)[-2]
	# Load the image and pre-process
	image = load_img(images_path, target_size=(image_sizeX, image_sizeY))
	image = img_to_array(image)
	image = preprocess_input(image)
	# Set (update) data and label in the list
	data.append(image)
	labels.append(label)

# Converts data and labels into NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# One-hot coding on labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Model training, with 80% of the dataset for training and 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# Build the training image generator for data augmentation
generator = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# Load the MobileNetV2 network, to ensure that the main FC layer sets are left out
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(image_sizeX, image_sizeY, 3)))

# Build the model head that will be placed on top of the base model
head_model = baseModel.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation="softmax")(head_model)

# Placing the main model (head) over the base model (this will be the real model that we will train)
model = Model(inputs=baseModel.input, outputs=head_model)

# Scans all layers in the base model and freezes them so they are not updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# Compile the new model
print("[INFO] Compiling the new model")
optimizer = Adam(lr=starting_learning, decay=starting_learning / training_size)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train the head
print("[INFO] Training to identify the head...")
head = model.fit(generator.flow(trainX, trainY, batch_size=batch_size), steps_per_epoch=len(trainX) // batch_size, validation_data=(testX, testY), validation_steps=len(testX) // batch_size, epochs=training_size)

# Make predictions in the test suite
print("[INFO] Verification of the neural network -> Making predictions with the test set")
predIdxs = model.predict(testX, batch_size=batch_size)

# For each image in the test suite, need to find the index of the label with the highest probability of prediction
predIdxs = np.argmax(predIdxs, axis=1)

# Print the rating report
print("[INFO] Rating report:")
print(classification_report(testY.argmax(axis=1), predIdxs,	target_names=lb.classes_))

# Save the model
print("[INFO] Saving the treined model...")
model_path = "trainer-model/mask_detector.model"
model.save(model_path, save_format="h5")
print("[INFO] Saved file: " + model_path)

# Plot loss and training accuracy
print("[INFO] Generating training graph...")
output_graph_path = "training-plot.png"
N = training_size
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), head.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), head.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), head.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), head.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Training cycle (Epoch)")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(output_graph_path)
print("[INFO] Saved file: " + output_graph_path)
