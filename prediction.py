# import os
# import kagglehub
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# import numpy as np
# import matplotlib.pyplot as plt



# # 🚀 Step 1: Download the Dataset
# dataset_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

# # Define directories
# base_dir = os.path.join(dataset_path, "chest_xray")
# train_dir = os.path.join(base_dir, "train")
# val_dir = os.path.join(base_dir, "val")
# test_dir = os.path.join(base_dir, "test")

# # 🚀 Step 2: Data Preprocessing
# img_size = 150  # Resize images to 150x150

# train_datagen = ImageDataGenerator(
#     rescale=1.0/255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )

# test_datagen = ImageDataGenerator(rescale=1.0/255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir, target_size=(img_size, img_size), batch_size=32, class_mode='binary'
# )

# val_generator = test_datagen.flow_from_directory(
#     val_dir, target_size=(img_size, img_size), batch_size=32, class_mode='binary'
# )

# test_generator = test_datagen.flow_from_directory(
#     test_dir, target_size=(img_size, img_size), batch_size=32, class_mode='binary', shuffle=False
# )

# # 🚀 Step 3: Build the CNN Model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
#     MaxPooling2D(2, 2),

#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),

#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),

#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')  # Output 1 neuron (Binary Classification)
# ])

# model.compile(optimizer=Adam(learning_rate=0.001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# # 🚀 Step 4: Train the Model
# epochs = 10
# history = model.fit(train_generator, validation_data=val_generator, epochs=epochs)

# # 🚀 Step 5: Evaluate the Model
# test_loss, test_acc = model.evaluate(test_generator)
# print(f"Test Accuracy: {test_acc:.2f}")

# # 🚀 Step 6: Save the Model
# model.save("pneumonia_cnn_model.h5")

