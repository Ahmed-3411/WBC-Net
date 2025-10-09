import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.notebook import tqdm

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


np.random.seed(42)
tf.random.set_seed(42)


print("Setting up the dataset path...")



for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)

dataset_base_path = "/kaggle/input/white-blood-cells-dataset"
train_path = os.path.join(dataset_base_path, "Train")
test_path = os.path.join(dataset_base_path, "Test-A")
classes = ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]


print("Loading and preprocessing images...")


def load_images_from_directory(directory_path, classes):
    images = []
    labels = []
    image_paths = []
    img_size = 128  
    
    for label_idx, class_name in enumerate(classes):
        class_path = os.path.join(directory_path, class_name)
        
        if not os.path.exists(class_path):
            print(f"Warning: Path {class_path} does not exist")
            continue
            
        for img_file in tqdm(os.listdir(class_path), desc=f"Loading {class_name}"):
            img_path = os.path.join(class_path, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((img_size, img_size))
                images.append(np.array(img))
                labels.append(label_idx)
                image_paths.append(img_path)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels), image_paths


X_train_data, y_train_data, train_image_paths = load_images_from_directory(train_path, classes)
print(f"Loaded {len(X_train_data)} training images")


X_test_data, y_test_data, test_image_paths = load_images_from_directory(test_path, classes)
print(f"Loaded {len(X_test_data)} test images")


print("\nData Exploration:")

train_class_counts = np.bincount(y_train_data)
for i, count in enumerate(train_class_counts):
    if i < len(classes):
        print(f"{classes[i]}: {count} training images")


plt.figure(figsize=(10, 5))
sns.countplot(x=y_train_data)
plt.title('Training Data Class Distribution')
plt.xlabel('Class Index')
plt.xticks(range(len(classes)), classes, rotation=45)
plt.ylabel('Count')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))
for i, class_name in enumerate(classes):
    if i < len(classes):
        class_indices = np.where(y_train_data == i)[0]
        if len(class_indices) > 0:
            sample_idx = class_indices[0]
            plt.subplot(1, len(classes), i+1)
            plt.imshow(X_train_data[sample_idx])
            plt.title(class_name)
            plt.axis('off')
plt.tight_layout()
plt.show()


print("\nPreparing data for training...")


X_train = X_train_data / 255.0
X_test = X_test_data / 255.0


y_train = tf.keras.utils.to_categorical(y_train_data, len(classes))
y_test = tf.keras.utils.to_categorical(y_test_data, len(classes))


X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Training set: {X_train.shape[0]} images")
print(f"Validation set: {X_val.shape[0]} images")
print(f"Test set: {X_test.shape[0]} images")

print("\nSetting up data augmentation...")

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

plt.figure(figsize=(15, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    augmented = datagen.random_transform(X_train[0])
    plt.imshow(augmented)
    plt.title(f'Augmented {i+1}')
    plt.axis('off')
plt.tight_layout()
plt.show()

print("\nBuilding high accuracy CNN model...")

img_size = 128  

def build_optimized_cnn():
    model = Sequential([
       
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(img_size, img_size, 3)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        
       
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        
        Flatten(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(len(classes), activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


cnn_model = build_optimized_cnn()
cnn_model.summary()


callbacks = [
    ModelCheckpoint('best_wbc_model.keras', save_best_only=True, monitor='val_accuracy', mode='max'),
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
]


print("\nTraining high accuracy CNN model...")

history = cnn_model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=10, 
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)


print("\nEvaluating model on test set...")


test_loss, test_acc = cnn_model.evaluate(X_test, y_test, verbose=1)
print(f"CNN Model - Test Accuracy: {test_acc:.4f}")


print(f"\nGenerating detailed metrics for the model...")

y_pred = cnn_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)


print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=classes))


plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()


print("\nVisualizing training history...")

plt.figure(figsize=(12, 4))


plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


print("\nVisualizing model predictions...")


indices = np.random.choice(range(len(X_test)), size=10, replace=False)

plt.figure(figsize=(20, 10))
for i, idx in enumerate(indices):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[idx])
    plt.title(f"True: {classes[y_true_classes[idx]]}\nPred: {classes[y_pred_classes[idx]]}", 
              color=('green' if y_true_classes[idx] == y_pred_classes[idx] else 'red'))
    plt.axis('off')
plt.tight_layout()
plt.show()

cnn_model.save('wbc_classification_model.keras')
print("\nModel saved as 'wbc_classification_model.keras'")


def predict_wbc_image(image_path, model=cnn_model):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction, axis=1)[0]
    
    print(f"Predicted class: {classes[pred_class]} with {prediction[0][pred_class]:.2%} confidence")
    
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {classes[pred_class]}\nConfidence: {prediction[0][pred_class]:.2%}")
    plt.axis('off')
    plt.show()
    
    
    plt.figure(figsize=(10, 3))
    sns.barplot(x=classes, y=prediction[0])
    plt.title('Prediction Probabilities')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return classes[pred_class], prediction[0]


if len(test_image_paths) > 0:
    sample_idx = np.random.choice(range(len(test_image_paths)))
    print(f"\nSample prediction demonstration:")
    predict_wbc_image(test_image_paths[sample_idx])

print("\nProject completed successfully!")
