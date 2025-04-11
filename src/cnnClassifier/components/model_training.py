import os
import random
import shutil
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import KFold
from cnnClassifier.entity.config_entity import TrainingConfig
from sklearn.metrics import classification_report

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

        print(f"🌟 Final Count: Normal: {self.count_images(normal_dir)}, Tumor: {self.count_images(tumor_dir)}")

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

    def load_image_dataframe(self):
        data = []
        for label in ["normal", "tumor"]:
            class_dir = os.path.join(self.config.training_data, label)
            for img in os.listdir(class_dir):
                if img.lower().endswith(('png', 'jpg', 'jpeg')):
                    data.append({
                        "filepath": os.path.join(class_dir, img),
                        "label": label
                    })
        return pd.DataFrame(data)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def plot_roc_auc(self, y_true, y_probs, class_names, fold):
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - Fold {fold}')
        plt.legend(loc="lower right")
        plt.savefig(f"artifacts/roc_auc_fold{fold}.png")
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, class_names, fold):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix - Fold {fold}')
        plt.savefig(f"artifacts/confusion_matrix_fold{fold}.png")
        plt.close()

    def train(self):
        k = 5
        df = self.load_image_dataframe()
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        all_metrics = []
        class_names = ['normal', 'tumor']

        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            print(f"\n🔁 Training Fold {fold + 1}/{k}")

            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2
            )

            val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

            common_args = dict(
                x_col='filepath',
                y_col='label',
                target_size=self.config.params_image_size[:-1],
                batch_size=self.config.params_batch_size,
                class_mode='categorical'
            )

            train_gen = train_datagen.flow_from_dataframe(train_df, shuffle=True, **common_args)
            val_gen = val_datagen.flow_from_dataframe(val_df, shuffle=False, **common_args)

            self.get_base_model()

            history = self.model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=self.config.params_epochs,
                steps_per_epoch=train_gen.samples // train_gen.batch_size,
                validation_steps=val_gen.samples // val_gen.batch_size
            )

            y_true = val_gen.classes
            y_probs = self.model.predict(val_gen)
            y_pred = np.argmax(y_probs, axis=1)

            # Compute metrics
            loss = history.history['val_loss'][-1]
            accuracy = history.history['val_accuracy'][-1]
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            report = classification_report(y_true, y_pred, output_dict=True)
            ppv = report['weighted avg']['precision']
            fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
            auc_score = auc(fpr, tpr)

            self.plot_roc_auc(y_true, y_probs, class_names, fold + 1)
            self.plot_confusion_matrix(y_true, y_pred, class_names, fold + 1)

            fold_metrics = {
                "fold": fold + 1,
                "loss": loss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "ppv": ppv,
                "roc_auc": auc_score,
            }
            all_metrics.append(fold_metrics)

            fold_model_path = Path(str(self.config.trained_model_path).replace(".h5", f"_fold{fold + 1}.h5"))
            self.save_model(path=fold_model_path, model=self.model)
            print(f"✅ Saved model for Fold {fold + 1} at {fold_model_path}")

        shutil.copy2(fold_model_path, "model/model.h5")
        print("📦 Final model copied to: model/model.h5")

        pd.DataFrame(all_metrics).to_csv("artifacts/fold_metrics.csv", index=False)
        print("📊 Metrics saved to artifacts/fold_metrics.csv")
