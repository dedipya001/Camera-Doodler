import os
import numpy as np
import cv2

from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("models/new_model_whitebg.h5")

def segment_and_predict(image_path="mask_capture.png"):

    os.makedirs("Resized", exist_ok=True)
    os.makedirs("Normalized", exist_ok=True)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    inverted_thresh = cv2.bitwise_not(thresh)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(inverted_thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right based on x-coordinate
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    predictions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Check if the width or height is too small, indicating noise or artifacts
        if w < 10 or h < 10:
            continue

        # Determine which dimension (width or height) is greater
        if w > h:
            # Calculate padding for height to make the image square
            pad = (w - h) // 2
            # Pad top and bottom with white space
            padded_char = cv2.copyMakeBorder(
                thresh[y : y + h, x : x + w],
                pad + 25,
                pad + 25,
                25,
                25,
                cv2.BORDER_CONSTANT,
                value=255,
            )
        else:
            # Calculate padding for width to make the image square
            pad = (h - w) // 2
            # Pad left and right with white space
            padded_char = cv2.copyMakeBorder(
                thresh[y : y + h, x : x + w],
                25,
                25,
                pad + 25,
                pad + 25,
                cv2.BORDER_CONSTANT,
                value=255,
            )

        # Resize the padded character to 28x28
        resized_char = cv2.resize(padded_char, (28, 28))

        # Save the resized character for debugging
        cv2.imwrite(f"Resized/Resized{x}.png", resized_char)

        # Normalize the resized character
        normalized_char = resized_char / 255.0

        # Reshape the normalized character to (28, 28, 1)
        normalized_char = normalized_char.reshape(28, 28, 1)
        cv2.imwrite(f"Normalized/Normalized{x}.png", normalized_char * 255.0)

        # Make prediction using the model
        prediction = model.predict(np.array([normalized_char]))
        predicted_class = np.argmax(prediction, axis=-1)
        predictions.append(predicted_class[0])

    return predictions
