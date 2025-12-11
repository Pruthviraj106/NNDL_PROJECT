# NNDL_PROJECT


# **Pneumonia Detection Using Deep Learning (MobileNetV2 Transfer Learning)**

This project implements an automated **Pneumonia Detection System** using **Deep Learning** and **Transfer Learning** with **MobileNetV2**.
The model is trained on the Kaggle **Chest X-Ray Pneumonia Dataset** and classifies chest X-ray images as **PNEUMONIA** or **NORMAL**.
The notebook includes dataset downloading, preprocessing, model creation, training, evaluation, and prediction for uploaded images.

---

## 📝 **Project Overview**

Pneumonia is a serious respiratory illness requiring timely diagnosis. Chest X-rays are the most common diagnostic tool, but manual interpretation can be slow and error-prone.

In this project:

* A pretrained **MobileNetV2** model is used as the feature extractor.
* A custom classifier is added for binary classification.
* Data augmentation improves robustness.
* The model is evaluated and used to predict new X-ray images.

This demonstrates how **deep learning** can support healthcare decision-making.

---

## 📦 **Dataset**

The dataset is automatically downloaded using:

```python
kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
```

### **Dataset Structure**

```
chest_xray/
├── train/
│   ├── NORMAL
│   └── PNEUMONIA
├── val/
│   ├── NORMAL
│   └── PNEUMONIA
└── test/
    ├── NORMAL
    └── PNEUMONIA
```

This dataset contains both bacterial and viral pneumonia samples.

---

## 🖼️ **Data Preprocessing**

The notebook uses **ImageDataGenerator** for:

* Rescaling: `1./255`
* Rotation
* Zoom
* Shear
* Horizontal flipping
* Batch loading (32 images per batch)

Example:

```python
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)
```

This ensures the model becomes more generalizable.

---

## 🧠 **Model Architecture**

The project uses **MobileNetV2** with ImageNet weights:

```python
base_model = MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)
base_model.trainable = False
```

### **Custom Classification Layers**

```python
model = tf.keras.Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

### **Training**

* Optimizer: **Adam**
* Loss function: **Binary Crossentropy**
* Epochs: **20**
* Metric: **Accuracy**

---

## 📊 **Model Evaluation**

The model is tested on the **test dataset**:

```python
loss, accuracy = model.evaluate(test_data)
```

Outputs shown in notebook:

* **Test Loss**
* **Test Accuracy**

(You can update final accuracy here after finishing training.)

---

## 🔍 **Making Predictions on New Images**

The notebook supports uploading an X-ray image:

```python
uploaded = files.upload()
img_path = list(uploaded.keys())[0]
```

Workflow:

1. Upload X-ray image
2. Preprocess it to size 224×224
3. Predict Pneumonia or Normal
4. Print confidence score

Example output:

```
Prediction: Pneumonia
Confidence: 92.64%
```

---

## 📁 **Project Structure**

```
Pneumonia_Detection/
│
├── pneumonia_detection.ipynb
├── dataset/   (auto-downloaded using link)
├── models/    (optional saved models)
└── README.md
```

---




## 📉 **Limitations**

* Dependent on the quality of X-ray dataset
* Cannot replace radiologists; acts as decision support
* Transfer learning may miss rare pneumonia patterns

---

## 🧬 **Future Enhancements**

* Enable fine-tuning of deeper MobileNetV2 layers
* Add Grad-CAM heatmaps for explainability
* Try advanced architectures (DenseNet, EfficientNet)
* Deploy using Flask/Streamlit
* Convert to TFLite for mobile-based diagnosis apps

---

## 🏁 **Conclusion**

This project demonstrates how **Deep Learning** and **MobileNetV2 transfer learning** can be applied to medical imaging tasks like pneumonia detection.
The resulting model provides fast, reliable predictions and can serve as a valuable diagnostic support tool in healthcare.

---



