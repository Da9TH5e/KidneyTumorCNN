import os
import random
import shutil
import tensorflow as tf
import numpy as np
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.preprocess_dataset()
        self.get_base_model()

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

    def count_images(self, directory):
        return len([f for f in os.listdir(directory) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

    def preprocess_dataset(self):
        normal_dir = os.path.join(self.config.training_data, "normal")
        tumor_dir = os.path.join(self.config.training_data, "tumor")

        if not os.path.exists(normal_dir) or not os.path.exists(tumor_dir):
            raise ValueError("❌ 'normal' or 'tumor' folder not found inside the dataset directory!")

        normal_count = self.count_images(normal_dir)
        tumor_count = self.count_images(tumor_dir)

        print(f"📊 Current Image Count: Normal: {normal_count}, Tumor: {tumor_count}")

        if normal_count < 10000:
            self.augment_images_to_target(normal_dir, 10000, "normal")

        if tumor_count < 10000:
            self.augment_images_to_target(tumor_dir, 10000, "tumor")

        print(f"🎯 Final Count: Normal: {self.count_images(normal_dir)}, Tumor: {self.count_images(tumor_dir)}")

    def augment_images_to_target(self, directory, target_count, keyword):
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(('png', 'jpg', 'jpeg')) and keyword in f.lower()]
        current_count = len(image_files)

        if current_count >= target_count:
            print(f"✅ {directory} already has {current_count} images. No augmentation needed.")
            return

        print(f"🔄 Augmenting images in {directory}: {current_count} → {target_count} images...")

        aug = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40,             
            width_shift_range=0.2,         
            height_shift_range=0.2,        
            shear_range=0.2,               
            zoom_range=0.2,                
            horizontal_flip=True,          
            vertical_flip=True,            
            brightness_range=[0.7, 1.3],   
            channel_shift_range=50.0,      
            rescale=1./255,                
            fill_mode="nearest"            
        )

        while current_count < target_count:
            img_name = random.choice(image_files)
            img_path = os.path.join(directory, img_name)

            image = tf.keras.preprocessing.image.load_img(img_path)
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = np.expand_dims(image, axis=0)

            for batch in aug.flow(image, batch_size=1, save_to_dir=directory, save_prefix="aug", save_format="jpg"):
                current_count += 1
                break  

        print(f"✅ Augmentation complete: {directory} now has {current_count} images.")


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

