import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Define directories (use forward slashes or raw strings to avoid escape sequence issues)
train_images_dir = r'train/images'
train_labels_dir = r'train/labels'

valid_images_dir = r'valid/images'
valid_labels_dir = r'valid/labels'

test_images_dir = r'test/images'
test_labels_dir = r'test/labels'

# Function to load images
def load_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)  # Resize to target size
    image = image / 255.0  # Normalize the image
    return image

# Function to load labels from YOLO-style text files (first number is class label)
def load_label(label_path):
    with open(label_path, 'r') as file:
        label_data = file.read().strip().split()
        class_label = int(label_data[0])  # Assuming first value is the class label
    return class_label

# Function to load dataset (images and corresponding labels)
def load_dataset(image_dir, label_dir, target_size=(224, 224)):
    images = []
    labels = []
    
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Ensure you're loading images
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')  # Match image name with label

            image = load_image(image_path, target_size)
            label = load_label(label_path)

            images.append(image)
            labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)

    # Map labels to a continuous range starting from 0
    unique_labels = np.unique(labels)
    label_mapping = {label: index for index, label in enumerate(unique_labels)}
    labels = np.array([label_mapping[label] for label in labels])

    # Automatically detect the number of unique classes
    num_classes = len(unique_labels)
    print(f"Detected {num_classes} unique classes in the dataset.")
    
    # One-hot encode labels based on the number of classes
    labels = to_categorical(labels, num_classes=num_classes)
    
    return images, labels

# Load train, valid, and test datasets
train_images, train_labels = load_dataset(train_images_dir, train_labels_dir)
valid_images, valid_labels = load_dataset(valid_images_dir, valid_labels_dir)
test_images, test_labels = load_dataset(test_images_dir, test_labels_dir)

print(f"Train dataset: {train_images.shape}, Labels: {train_labels.shape}")
print(f"Valid dataset: {valid_images.shape}, Labels: {valid_labels.shape}")
print(f"Test dataset: {test_images.shape}, Labels: {test_labels.shape}")
# Create the model
from tensorflow.keras import layers, models

def create_model(input_shape=(224, 224, 3), num_classes=2):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # num_classes should match your dataset
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_model(num_classes=train_labels.shape[1])
model.summary()

# Train the model
EPOCHS = 20
BATCH_SIZE = 32

history = model.fit(train_images, train_labels,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(valid_images, valid_labels))

# Save the trained model
model.save('plastic_classifier_with_labels.h5')
