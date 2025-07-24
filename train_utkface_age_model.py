import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Parameters
img_dir = 'UTKFace'
img_size = (64, 64)

# 1. Load images and ages
def parse_age(filename):
    return int(filename.split('_')[0])

def load_utkface_images(img_dir, img_size, max_images=None):
    images = []
    ages = []
    files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    if max_images:
        files = files[:max_images]
    for fname in files:
        try:
            age = parse_age(fname)
            img_path = os.path.join(img_dir, fname)
            img = Image.open(img_path).convert('RGB').resize(img_size)
            images.append(np.array(img))
            ages.append(age)
        except Exception as e:
            print(f'Could not process {fname}: {e}')
    return np.array(images), np.array(ages)

print('Loading images...')
X, y = load_utkface_images(img_dir, img_size)
X = X / 255.0  # Normalize

y = y.astype(np.float32)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build CNN regression model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='linear')  # Regression output
])

model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])

# 3. Train
print('Training model...')
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# 4. Save model
model.save('age_regression_model.h5')
print('Model saved as age_regression_model.h5') 