import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import joblib
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
import cv2

data_dir = "dataset"  # Path to the dataset folder
saved_data_file = "saved_data.joblib"

def load_images(directory):
    flag=True
    print("Start")
    images = []
    labels = []
    label_dict = {
        "plus": 10,
        "minus": 11,
        "mul": 12,
        "div": 13,
    }  # Map symbols to numeric labels
    for label_folder in os.listdir(directory):
        print("Inside", label_folder)
        label_folder_path = os.path.join(directory, label_folder)
        if label_folder in ["plus", "minus", "mul", "div"]:  # Symbols
            numeric_label = label_dict[label_folder]
        else:  # Digits
            numeric_label = int(label_folder)

        for image_file in os.listdir(label_folder_path):
            image_path = os.path.join(label_folder_path, image_file)

            # pre process images
            image = Image.open(image_path).convert('L') #grayscale
            # image = Image.open(image_path)  # grayscale
            image = image.resize((28, 28))
            if flag:
                cv2.imwrite("Check.png",np.array(image))
                flag=False
            images.append(np.array(image))
            labels.append(numeric_label)
    return np.array(images), np.array(labels)


if os.path.exists(saved_data_file):
    print("Loading saved data...")
    images, labels = joblib.load(saved_data_file)
else:
    print("Loading data...")
    images, labels = load_images(data_dir)
    joblib.dump((images, labels), saved_data_file)

images = images / 255.0  # Normalize data

# Shuffle the data
images, labels = shuffle(images, labels, random_state=42)

train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

num_classes = 14  # 0-9 digits + 4 symbols

train_labels = to_categorical(train_labels, num_classes=num_classes)
test_labels = to_categorical(test_labels, num_classes=num_classes)

train_images = train_images.reshape(-1, 28, 28, 1)  # Reshape data to fit the model
test_images = test_images.reshape(-1, 28, 28, 1)

model = Sequential()

# Model layers
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

print("Model Copile")

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train
print("Training")
history = model.fit(
    train_images,
    train_labels,
    epochs=10,
    batch_size=32,
    validation_data=(test_images, test_labels),
)

# Plot training & validation accuracy
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()

# Saving the model
model.save(r"new_model.h5")
