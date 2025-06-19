# Install OpenCV (you only need this once in Colab)
!pip install opencv-python

# Import all the required libraries
import cv2                          # For working with images and face detection
import numpy as np                 # For numerical operations
import pandas as pd                # For handling the CSV dataset
import matplotlib.pyplot as plt    # For displaying the image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load the FER2013 dataset from Kaggle (make sure you've uploaded it to Colab)
data = pd.read_csv('/content/fer2013.csv')
data.head()  # Just to preview the first few rows

# These are the emotion labels the model will predict
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# We'll now extract images and labels from the CSV
X = []  # Will hold all the image data
y = []  # Will hold all the emotion labels

# Loop through each row in the CSV file
for index, row in data.iterrows():
    pixels = row['pixels'].split()  # Get the pixel values as a list of strings
    if len(pixels) != 2304:         # Skip the row if the image data is corrupted
        continue
    image = np.array(pixels, dtype='float32').reshape(48, 48, 1)  # Reshape to 48x48x1
    X.append(image)                 # Add the image to our list
    y.append(row['emotion'])       # Add the label

# Convert lists into numpy arrays and normalize image pixel values (0 to 1)
X = np.array(X) / 255.0
y = to_categorical(y, num_classes=7)  # Convert labels into one-hot encoded format

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optional: count how many invalid rows were skipped
invalid = data['pixels'].apply(lambda x: len(x.split()) != 2304).sum()
print(f"Invalid rows found: {invalid}")

# Build the Convolutional Neural Network (CNN) model
model = Sequential([
    # First convolutional layer + max pooling
    Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Dropout(0.25),  # Dropout to prevent overfitting

    # Second convolutional layer + max pooling
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Flatten the image and add fully connected layers
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Dropout again
    Dense(7, activation='softmax')  # Final output layer for 7 emotion classes
])

# Compile the model: use Adam optimizer and categorical cross-entropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()  # Show the structure of the model

# Train the model on the dataset (35 epochs, batch size 64)
model.fit(X_train, y_train, epochs=35, batch_size=64, validation_data=(X_test, y_test))

# ---- PREDICTION ON CUSTOM IMAGE ---- #

# Load your own image from file (/sample_images/)
img = cv2.imread('/sample_images/sad.jpg')

# Convert it to grayscale (FER2013 model expects grayscale input)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# For each face found, predict the emotion
for (x, y, w, h) in faces:
    face = gray[y:y+h, x:x+w]               # Crop the face
    face = cv2.resize(face, (48, 48))       # Resize to 48x48 as expected by model
    face = face.astype("float32") / 255.0   # Normalize pixel values
    face = np.expand_dims(face, axis=0)     # Add batch dimension
    face = np.expand_dims(face, axis=-1)    # Add channel dimension (grayscale)

    # Predict emotion using the trained model
    prediction = model.predict(face)
    emotion_index = np.argmax(prediction)
    emotion_label = emotion_labels[emotion_index]

    # Draw a rectangle and label on the original image
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(img, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)

# Convert image from BGR (OpenCV format) to RGB (matplotlib format)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image with the predicted emotion and bounding box
plt.figure(figsize=(6,6))
plt.imshow(img_rgb)
plt.title(f"Detected Emotion: {emotion_label}")
plt.axis('off')
plt.show()