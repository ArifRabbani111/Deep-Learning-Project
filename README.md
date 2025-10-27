# 🧠 MNIST CNN Model with Hyperparameter Tuning

This project demonstrates a **complete deep learning workflow** for classifying handwritten digits from the **MNIST dataset** using a **Convolutional Neural Network (CNN)**.  
It covers every stage — from data preprocessing to hyperparameter optimization — to achieve a high-performing and well-regularized model.

---

## 🚀 Project Overview

The objective of this notebook is to **build, train, evaluate, and optimize** a CNN model capable of recognizing handwritten digits (0–9) from grayscale 28×28 pixel images.  
Through systematic training and tuning, the model’s performance is enhanced for better accuracy and generalization.

---

## 🧩 Steps Performed

### **1. Load the MNIST Dataset**
- Loaded using TensorFlow’s built-in `tf.keras.datasets.mnist`.
- Split into **60,000 training** and **10,000 testing** images.

### **2. Data Preprocessing**
- Normalized pixel values to the range `[0, 1]`.
- Reshaped images to `(28, 28, 1)` to fit CNN input dimensions.
- Converted labels into integer-encoded classes.

### **3. Build the CNN Model**
- Defined a **baseline CNN** using `tensorflow.keras.Sequential`.
- Layers included:
  - `Conv2D` and `MaxPooling2D` for feature extraction.
  - `Flatten` and `Dense` for classification.
- Designed to balance simplicity, performance, and interpretability.

### **4. Model Compilation**
- **Optimizer:** Adam  
- **Loss Function:** Sparse Categorical Crossentropy  
- **Metric:** Accuracy  

### **5. Train and Validate the Model**
- Trained the model on the training set and validated on the test set.
- Visualized accuracy and loss trends across epochs.

### **6. Evaluate and Visualize Results**
- Evaluated model performance on unseen test data.
- Displayed **predictions vs. actual labels** for sample images.
- Provided insights into strengths and misclassifications.

---

## 🔍 Hyperparameter Tuning

To further improve performance, **Keras Tuner’s Random Search** was used to explore various configurations of the CNN model.

### **Tuning Strategy**
- Defined a hyperparameter search space for:
  - Number of filters and kernel sizes in `Conv2D` layers
  - Learning rate of the optimizer
  - Dropout rate for regularization
  - Batch size and dense layer units
- Added **Batch Normalization** and **Dropout** layers for better regularization.
- Introduced an additional `Conv2D` layer to explore deeper networks.

### **Tuning Process**
- Implemented a modular function `build_cnn_model(hp)` to dynamically construct models based on hyperparameters.
- Used **RandomSearch** to identify the optimal configuration based on validation accuracy.

### **Best Model Selection**
- Retrieved best hyperparameters from the tuner.
- Rebuilt and retrained the CNN using the optimal configuration.
- Evaluated on the full training and test datasets.

---

## 🧾 Results and Insights

- The tuned CNN achieved **significant improvement in accuracy** and generalization.
- Both models were compared in terms of training behavior and evaluation metrics.

**Key Findings:**
- **Regularization** (Dropout, BatchNorm) effectively reduced overfitting.  
- **Learning rate tuning** stabilized convergence.  
- **Model depth** enhanced feature extraction without vanishing gradients.

---

## 🧮 Technologies Used
- 🐍 Python  
- 🧠 TensorFlow / Keras  
- 🎯 Keras Tuner  
- 📊 NumPy, Matplotlib, Seaborn for data analysis and visualization  

---

## 📊 Performance Summary

| Model Version | Test Accuracy | Validation Loss | Key Features |
|----------------|---------------|------------------|---------------|
| Baseline CNN | ~98% | Moderate | Basic CNN architecture |
| Tuned CNN | **>99%** | Lower | BatchNorm, Dropout, optimized hyperparams |

---

## 🧠 Key Learnings
- Systematic hyperparameter tuning greatly boosts CNN performance.  
- Regularization is critical for preventing overfitting.  
- Visualization helps identify convergence issues and failure patterns.  
- The MNIST dataset remains a powerful benchmark for experimentation.

---

## 📌 Future Enhancements
- Extend to **Fashion-MNIST** or **CIFAR-10** for more complex tasks.  
- Use **Bayesian Optimization** or **Hyperband** for efficient tuning.  
- Visualize **feature maps** to interpret learned representations.  
- Apply **transfer learning** or **quantization** for deployment efficiency.

---

---

## 🏁 Conclusion

This project showcases the **complete deep learning pipeline** — from building a baseline CNN to performing **rigorous hyperparameter optimization**.  
The final tuned model delivers high accuracy on the MNIST dataset and serves as a foundation for more advanced **computer vision research**.

---

### 📜 Author
**ARIF RABBANI**  
Software Engineering Student | Machine Learning Enthusiast  





