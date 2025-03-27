# import os
# import shutil
# import urllib.request as request
# from zipfile import ZipFile
# import tensorflow as tf
# import time
# from pathlib import Path
# from cnnClassifier.entity.config_entity import TrainingConfig


# class Training:
#     def __init__(self, config: TrainingConfig):
#         self.config = config

    
#     def get_base_model(self):
#         self.model = tf.keras.models.load_model(
#             self.config.updated_base_model_path
#         )

#     def train_valid_generator(self):

#         datagenerator_kwargs = dict(
#             rescale = 1./255,
#             validation_split=0.20
#         )

#         dataflow_kwargs = dict(
#             target_size=self.config.params_image_size[:-1],
#             batch_size=self.config.params_batch_size,
#             interpolation="bilinear"
#         )

#         valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
#             **datagenerator_kwargs
#         )

#         self.valid_generator = valid_datagenerator.flow_from_directory(
#             directory=self.config.training_data,
#             subset="validation",
#             shuffle=False,
#             **dataflow_kwargs
#         )

#         if self.config.params_is_augmentation:
#             train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
#                 rotation_range=40,
#                 horizontal_flip=True,
#                 width_shift_range=0.2,
#                 height_shift_range=0.2,
#                 shear_range=0.2,
#                 zoom_range=0.2,
#                 **datagenerator_kwargs
#             )
#         # if self.config.params_is_augmentation:
#         #     train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
#         #     rotation_range=40,
#         #     horizontal_flip=True,
#         #     vertical_flip=True,
#         #     width_shift_range=0.2,
#         #     height_shift_range=0.2,
#         #     shear_range=0.2,
#         #     zoom_range=0.2,
#         #     brightness_range=[0.8, 1.2],  # Brightness variation
#         #     fill_mode="nearest",
#         #     rescale=1./255,
#         #     validation_split=0.20
#         # )
#         else:
#             train_datagenerator = valid_datagenerator

#         self.train_generator = train_datagenerator.flow_from_directory(
#             directory=self.config.training_data,
#             subset="training",
#             shuffle=True,
#             **dataflow_kwargs
#         )

    
#     @staticmethod
#     def save_model(path: Path, model: tf.keras.Model):
#         model.save(path)



    
#     def train(self):
#         self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
#         self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

#         self.model.fit(
#             self.train_generator,
#             epochs=self.config.params_epochs,
#             steps_per_epoch=self.steps_per_epoch,
#             validation_steps=self.validation_steps,
#             validation_data=self.valid_generator
#         )
#         # self.model.fit(
#         #     self.train_generator,
#         #     epochs=self.config.params_epochs,
#         #     steps_per_epoch=self.steps_per_epoch,
#         #     validation_steps=self.validation_steps,
#         #     validation_data=self.valid_generator,
#         #     class_weight=class_weights  # Add class weights here
#         # )

#         self.save_model(
#             path=self.config.trained_model_path,
#             model=self.model
#         )

#         os.makedirs("model", exist_ok=True)

#         # Copy trained model to 'model' directory
#         shutil.copy2("artifacts/training/model.h5", "model/model.h5")
# # import os
# # import shutil
# # import tensorflow as tf
# # from pathlib import Path
# # from cnnClassifier.entity.config_entity import TrainingConfig


# # class Training:
# #     def __init__(self, config: TrainingConfig):
# #         self.config = config

# #     def get_base_model(self):
# #         """Load the pre-trained model."""
# #         self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

# #     def train_valid_generator(self):
# #         """Create training and validation data generators."""
# #         datagenerator_kwargs = dict(
# #             rescale=1.0 / 255, validation_split=0.20
# #         )

# #         dataflow_kwargs = dict(
# #             target_size=self.config.params_image_size[:-1],
# #             batch_size=self.config.params_batch_size,
# #             interpolation="bilinear",
# #         )

# #         # Validation data generator (NO augmentation)
# #         valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
# #             **datagenerator_kwargs
# #         )

# #         self.valid_generator = valid_datagenerator.flow_from_directory(
# #             directory=self.config.training_data,
# #             subset="validation",
# #             shuffle=False,
# #             **dataflow_kwargs
# #         )

# #         # Training data generator (WITH augmentation)
# #         # if self.config.params_is_augmentation:
# #         #     train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
# #         #         rotation_range=40,
# #         #         horizontal_flip=True,
# #         #         vertical_flip=True,
# #         #         width_shift_range=0.2,
# #         #         height_shift_range=0.2,
# #         #         shear_range=0.2,
# #         #         zoom_range=0.2,
# #         #         brightness_range=[0.8, 1.2],  # Brightness variation
# #         #         fill_mode="nearest",
# #         #         validation_split=0.20,
# #         #         **datagenerator_kwargs  # Ensuring rescale is applied correctly
# #         #     )
# #         if self.config.params_is_augmentation:
# #             train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
# #                 rotation_range=40,
# #                 horizontal_flip=True,
# #                 vertical_flip=True,
# #                 width_shift_range=0.2,
# #                 height_shift_range=0.2,
# #                 shear_range=0.2,
# #                 zoom_range=0.2,
# #                 brightness_range=[0.8, 1.2],  # Brightness variation
# #                 fill_mode="nearest",
# #                 **datagenerator_kwargs  # Use kwargs here to avoid duplicate validation_split
# #             )

# #         else:
# #             train_datagenerator = valid_datagenerator  # Use same generator if no augmentation

# #         self.train_generator = train_datagenerator.flow_from_directory(
# #             directory=self.config.training_data,
# #             subset="training",
# #             shuffle=True,
# #             **dataflow_kwargs
# #         )

# #     @staticmethod
# #     def save_model(path: Path, model: tf.keras.Model):
# #         """Save the trained model."""
# #         path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
# #         model.save(path)

# #     def train(self):
# #         """Train the model with class weights handling imbalance."""
# #         self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
# #         self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

# #         # Compute class weights to handle imbalance
# #         class_weights = self.compute_class_weights()

# #         # Train the model
# #         self.model.fit(
# #             self.train_generator,
# #             epochs=self.config.params_epochs,
# #             steps_per_epoch=self.steps_per_epoch,
# #             validation_steps=self.validation_steps,
# #             validation_data=self.valid_generator,
# #             class_weight=class_weights,  # Fix: Properly define and use class weights
# #         )

# #         # Save the trained model
# #         self.save_model(path=self.config.trained_model_path, model=self.model)

# #         # Copy trained model to 'model' directory
# #         os.makedirs("model", exist_ok=True)
# #         shutil.copy2(self.config.trained_model_path, "model/model.h5")

# #     def compute_class_weights(self):
# #         """Calculate class weights for handling imbalanced data."""
# #         from sklearn.utils.class_weight import compute_class_weight
# #         import numpy as np

# #         # Get class indices from the training generator
# #         class_labels = list(self.train_generator.class_indices.values())

# #         # Count samples per class
# #         class_counts = np.bincount(self.train_generator.classes)
        
# #         # Compute class weights
# #         class_weights = compute_class_weight(
# #             class_weight="balanced",
# #             classes=np.array(class_labels),
# #             y=self.train_generator.classes
# #         )

# #         # Convert to dictionary
# #         return {i: weight for i, weight in enumerate(class_weights)}
import os
import shutil
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)



    
    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

        os.makedirs("model", exist_ok=True)

        # Copy trained model to 'model' directory
        shutil.copy2("artifacts/training/model.h5", "model/model.h5")