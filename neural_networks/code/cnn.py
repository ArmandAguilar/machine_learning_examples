import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.preprocessing import image
import matplotlib.pyplot as plt
import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# Download the MNIST dataset
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# Check the shape of the training data
print("Shape of training data:", train_data.shape)

# Plot an example image and label
plt.imshow(train_data[55], cmap='gray')
plt.title("Label: " + str(train_labels[55]))
plt.show()

# Normalize and reshape the data
x_train = train_data.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_test = test_data.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

# One-hot encode the labels
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

# Create a CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy:", test_accuracy)

# Load a imgae
img_path = '../data/numbers/5.jpg'  # Path of image in our HD
img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')

# Check the image send it to RNN
plt.imshow(img, cmap='gray')
plt.title("Image Loaded")
plt.show()

#convert the img in a array with numpy and processing
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # Normalize img
img_array = img_array.reshape((1, 28, 28, 1))  # Re-shape the image

# Make the prediction of image
predictions = model.predict(img_array)

# Get the prediction of RNN
predicted_label = np.argmax(predictions)
print("The number is: " + str(predicted_label))