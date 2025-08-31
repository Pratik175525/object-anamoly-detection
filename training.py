import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# ==========================
# Dataset Paths
# ==========================
train_dir = "dataset/train"  # contains 'good' and 'defective'
test_dir = "dataset/test"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# ==========================
# Data Preprocessing
# ==========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# ==========================
# CNN Model
# ==========================
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()

# ==========================
# Train the Model
# ==========================
EPOCHS = 15
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# ==========================
# Save the Model
# ==========================
model.save("keras_model.h5")
print("Model saved as .h5")

# ==========================
# Convert to TensorFlow Lite
# ==========================
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model_unquant.tflite", "wb") as f:
    f.write(tflite_model)

print(" Model converted and saved as model_unquant.tflite")
