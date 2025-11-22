# 🩺 Kidney Tumor Classification (Deep Learning · CNN)

This project builds an automated kidney tumor classification system using deep learning.  
The goal is to classify CT scan images into categories such as **Normal**, **Cyst**, **Stone**, or **Tumor**.

## 🚀 Features
- Preprocessed CT scan images (resizing, normalization, augmentation)
- Custom CNN / Transfer Learning (ResNet50 / EfficientNet)
- High training accuracy with regularization
- Confusion matrix & classification report
- Model saved in `.h5` format for deployment

## 📂 Dataset
The dataset consists of labeled kidney CT scan images.  
Each folder represents a class:

/data
/normal
/tumor


## 🔧 Setup
```bash
pip install tensorflow keras numpy matplotlib scikit-learn

🛠️ Training the Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

train_gen = ImageDataGenerator(rescale=0.001, validation_split=0.2)
train = train_gen.flow_from_directory("data", target_size=(224,224),
                                      batch_size=32, subset='training')
val = train_gen.flow_from_directory("data", target_size=(224,224),
                                    batch_size=32, subset='validation')

base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False

model = models.Sequential([
    base,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train, validation_data=val, epochs=10)
model.save("kidney_tumor_model.h5")

📈 Results

    Accuracy: ~90% (varies by dataset)

    Misclassifications analyzed via confusion matrix

📦 Deployment

The final .h5 model can be:

    Served with Flask/FastAPI

    Deployed to AWS EC2 / Lambda

    Integrated in a medical diagnostic support tool

👥 Team

    A team of 5 members worked on this project.

    My role: Model optimization + AWS deployment (not completed due to technical issues).

📝 License

MIT License


---

If you want, I can also create:
⭐ A longer professional README  
⭐ Add architecture diagrams, dataset explanation, or metrics  
⭐ Make it specific to your team + contribution + AWS hosting issue
