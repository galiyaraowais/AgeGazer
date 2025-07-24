import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# SETTINGS
dataset_dir = "UTKFace"
image_size = (224, 224)
max_images = 3000  # Change to 5000+ later when stable
epochs = 10
batch_size = 32

# LOAD DATA
X, y = [], []
print("Loading images...")
for i, filename in enumerate(os.listdir(dataset_dir)):
    if filename.endswith(".jpg"):
        try:
            age = int(filename.split("_")[0])
            img_path = os.path.join(dataset_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, image_size)
            X.append(img)
            y.append(age)
        except:
            continue
    if len(X) >= max_images:
        break

X = np.array(X, dtype='float32') / 255.0
y = np.array(y, dtype='float32')
print(f"Loaded {len(X)} images.")

# SPLIT
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# MODEL
base = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
x = base.output
x = GlobalAveragePooling2D()(x)
pred = Dense(1, activation='linear')(x)
model = Model(inputs=base.input, outputs=pred)

# Freeze base model layers
for layer in base.layers:
    layer.trainable = False

# COMPILE & TRAIN
model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
checkpoint = ModelCheckpoint("custom_age_model.h5", monitor='val_loss', save_best_only=True)

print("Training started...")
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])

print("✅ Done! Model saved as 'custom_age_model.h5'")
