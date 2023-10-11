# USAGE
# python predict.py --input output/test_images.txt

# import the necessary packages
import config
import keras
from tensorflow.keras.preprocessing.image import img_to_array
from keras.utils import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import cv2
import os

curd = os.path.dirname(os.path.abspath(__file__)) + '\\'

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default=curd+'dataset/images',
	help="path to input image/text file of image filenames")
args = vars(ap.parse_args())

input_path = args['input']
imagePaths = []
for path in os.listdir(input_path):
    if os.path.isfile(os.path.join(input_path, path)):
        imagePaths.append(input_path + '//' + path)

# load our trained bounding box regressor from disk
print("[INFO] loading object detector...")
model = load_model(config.MODEL_PATH)

# loop over the images that we'll be testing using our bounding box
# regression model
for imagePath in imagePaths:
	# load the input image (in Keras format) from disk and preprocess
	# it, scaling the pixel intensities to the range [0, 1]
	image = keras.utils.load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image) / 255.0
	image = np.expand_dims(image, axis=0)	# (1,224,244,3)

	# make bounding box predictions on the input image
	preds = model.predict(image)	# (1,4)
	preds = model.predict(image)[0]
	(startX, startY, endX, endY) = preds

	# load the input image (in OpenCV format), resize it such that it
	# fits on our screen, and grab its dimensions
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# scale the predicted bounding box coordinates based on the image
	# dimensions
	startX = int(startX * w)
	startY = int(startY * h)
	endX = int(endX * w)
	endY = int(endY * h)

	# draw the predicted bounding box on the image
	cv2.rectangle(image, (startX, startY), (endX, endY),
		(0, 255, 0), 2)

	# show the output image
	cv2.imshow("Output", image)
	cv2.waitKey(0)