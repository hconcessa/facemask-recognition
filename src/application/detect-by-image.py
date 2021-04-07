from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os


# Load the face detector model
print("[INFO] Loading faces detector model...")
prototxtPath = os.path.sep.join(["trainer-model/face/deploy.prototxt"])
weightsPath = os.path.sep.join(["trainer-model/face/res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the face mask detector model from disk
print("[INFO] Loading mask detector model...")
model = load_model("trainer-model/mask/mask_detector.model")

# Load image and get spacial dimensions
imageExamplePath = 'examples/1.png'
print("[INFO] Loading image from " + imageExamplePath)
image = cv2.imread(imageExamplePath)
orige = image.copy()
(h, w) = image.shape[:2]

# constructor a blob by image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
# Pass the blob from network and get the faces detected
print("[INFO] Computing face detector...")
net.setInput(blob)
detecteds = net.forward()

# Text to bounding box
strWithMask = "With Mask"
strWithoutMask = "Without Mask"

# Loop in all faces detected
for i in range(0, detecteds.shape[2]):
	# Extract the confidence (probability) associated with the face detected
	probability = detecteds[0, 0, i, 2]
	# Filters out the low chance of hit detections. They pass only with more than 50% chance of success
	if probability > 0.5:
		# Calculate the coordinates (x, y) around the detected object
		box = detecteds[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		# Checking that the bounding boxes are within the dimensions of the frame
		(startX, startY) = (max(0, startX), max(0, startY))
		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
		# Extracts the ROI from the face and converts the image from BGR to RGB
		face = image[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # Resize to 224x224 and pre process
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)
		# Pass the detected face in mask detector model
		(mask, withoutMask) = model.predict(face)[0]
		# Set color and text to paint in bounding box
		label = strWithMask if mask < withoutMask else strWithoutMask
		color = (0, 255, 0) if label == strWithMask else (0, 0, 255)
		# Set the propability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		# show the label and the rectangle in the output image
		cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# Show final image
print("[INFO] Opening image in new window")
cv2.imshow("Mask detector -> github/hconcessa", image)
cv2.waitKey(0)