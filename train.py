import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224

train_data = ImageDataGenerator(
rescale=1./255,
validation_split=0.2
)

train_generator = train_data.flow_from_directory(
"dataset",
target_size=(224,224),
batch_size=32,
class_mode="categorical",
subset="training"
)

val_generator = train_data.flow_from_directory(
"dataset",
target_size=(224,224),
batch_size=32,
class_mode="categorical",
subset="validation"
)

base_model=tf.keras.applications.ResNet50(
weights='imagenet',
include_top=False,
input_shape=(224,224,3)
)

base_model.trainable=False

model=tf.keras.Sequential([
base_model,
tf.keras.layers.GlobalAveragePooling2D(),
tf.keras.layers.Dense(3,activation='softmax')
])

model.compile(
optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy']
)

model.fit(
train_generator,
validation_data=val_generator,
epochs=10
)

model.save("model.h5")