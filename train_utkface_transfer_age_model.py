import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parameters
img_dir = 'UTKFace'
img_size = (96, 96)  # MobileNetV2 default is 96x96 or larger
batch_size = 32

def parse_age(filename):
    return int(filename.split('_')[0])

def load_utkface_images(img_dir, img_size, max_images=None):
    images = []
    ages = []
    files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    if max_images:
        files = files[:max_images]
    for i, fname in enumerate(files):
        try:
            age = parse_age(fname)
            img_path = os.path.join(img_dir, fname)
            img = Image.open(img_path).convert('RGB').resize(img_size)
            images.append(np.array(img))
            ages.append(age)
            if i % 500 == 0:
                print(f"Loaded {i} images...")
        except Exception as e:
            print(f'Could not process {fname}: {e}')
    return np.array(images), np.array(ages)

print('Loading images...')
X, y = load_utkface_images(img_dir, img_size)
X = X / 255.0  # Normalize

y = y.astype(np.float32)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8,1.2]
)
val_datagen = ImageDataGenerator()

train_gen = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_gen = val_datagen.flow(X_val, y_val, batch_size=batch_size)

# Build transfer learning model
base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model

inputs = Input(shape=img_size + (3,))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation='linear')(x)
model = Model(inputs, outputs)

model.compile(optimizer=Adam(learning_rate=1e-3), loss='mean_squared_error', metrics=['mae'])

# Train
print('Training model (transfer learning)...')
history = model.fit(
    train_gen,
    epochs=15,
    validation_data=val_gen
)

# Optionally unfreeze and fine-tune
print('Fine-tuning base model...')
base_model.trainable = True
model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mae'])
history_finetune = model.fit(
    train_gen,
    epochs=5,
    validation_data=val_gen
)

# Save model
model.save('age_transfer_model.h5')
print('Model saved as age_transfer_model.h5') 