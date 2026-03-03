import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# ==========================================
# 1. Configuration and Paths
# ==========================================
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 20
BASE_DIR = 'dataset'  # Put your dataset folder here (with train/val/test subfolders)

# Check if dataset directory exists, otherwise create placeholders
if not os.path.exists(BASE_DIR):
    print(f"Dataset directory '{BASE_DIR}' not found. Please create it with 'train', 'validation', and 'test' subfolders containing class folders.")
    # Exit or continue for demonstration. We continue to show the structure.

# ==========================================
# 2. Data Augmentation (Increasing Dataset Size)
# ==========================================
print("Setting up Data Augmentation...")
# This significantly increases the effective dataset size by applying random transformations
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation and Test data should not be augmented (only rescaled)
test_datagen = ImageDataGenerator(rescale=1./255)

# Example flow_from_directory (Requires actual dataset to run fully)
try:
    train_generator = train_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    validation_generator = test_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'validation'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False # Keep false for testing
    )
    
    num_classes = train_generator.num_classes
    class_indices = train_generator.class_indices
    # Save class indices
    with open('plant_disease_classes.json', 'w') as f:
        json.dump({v: k for k, v in class_indices.items()}, f)
        
except FileNotFoundError:
    print("Dataset folders not found. Using dummy generators for code structure demonstration.")
    num_classes = 39 # Default for this project
    train_generator = None


# ==========================================
# 3. Model Building (CNN)
# ==========================================
print("Building CNN Model...")
# Option A: Custom Deep CNN
def build_custom_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Option B: Transfer Learning with MobileNetV2 (Recommended for better accuracy)
def build_transfer_learning_model(input_shape, num_classes):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False # Freeze base model initially
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Choose model
model = build_transfer_learning_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# ==========================================
# 4. Training
# ==========================================
if train_generator is not None:
    print("Starting Training...")
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint('models/best_cnn_model.keras', save_best_only=True)
    ]

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # Plot training results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    plt.savefig('training_history.png')
    plt.show()


# ==========================================
# 5. Testing Code (Evaluation)
# ==========================================
print("==========================================")
print("TESTING THE MODEL")
print("==========================================")
if train_generator is not None and test_generator is not None:
    # Load the best model
    # model = tf.keras.models.load_model('models/best_cnn_model.keras')
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Predict on a single batch
    x_test, y_test = next(test_generator)
    predictions = model.predict(x_test)
    
    # Show predictions for the first few images
    class_labels = {v: k for k, v in class_indices.items()}
    for i in range(min(5, len(x_test))):
        true_label = class_labels[np.argmax(y_test[i])]
        pred_label = class_labels[np.argmax(predictions[i])]
        confidence = np.max(predictions[i]) * 100
        print(f"Image {i+1}: True: {true_label} | Predicted: {pred_label} | Confidence: {confidence:.2f}%")
else:
    print("Test data generator not available. Here is an example to evaluate a single test image.")
    def test_single_image(model, img_path):
        if not os.path.exists(img_path):
            print(f"Image file {img_path} not found.")
            return
            
        try:
            img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0 # Rescale
            
            prediction = model.predict(img_array)
            top_index = np.argmax(prediction[0])
            confidence = prediction[0][top_index] * 100
            print(f"Predicted class index: {top_index}, Confidence: {confidence:.2f}%")
        except Exception as e:
            print(f"Could not evaluate test image '{img_path}'. Error: {e}")

    # Example test
    test_single_image(model, 'test_img.jpg')

# Save final model
model.save('models/final_cnn_model.keras')
print("Model saved to models/final_cnn_model.keras")
