# 🏡 California Housing Price Prediction with Deep Learning

An end-to-end regression pipeline built with **TensorFlow / Keras** to predict California median home values using demographic, economic, and geographic features.

---

## 📌 Project Overview

This project implements a Multi-Layer Perceptron (MLP) neural network on the classic **California Housing Dataset**. 

* **Task:** Continuous Regression (predicting median house value).
* **Dataset Size:** 20,640 block groups / samples.
* **Target Units:** Continuous value in units of \$100,000 (e.g., `2.5` = \$250,000).

---

## 📊 Key Results

| Metric | Score | Practical Interpretation |
| :--- | :--- | :--- |
| **Test MAE** | **~0.3522** | Average estimate is off by **~$33,000 – $36,000** on homes averaging ~$207,000. |
| **$R^2$ Score** | **~0.7867** | Model explains **~80%** of the total price variance. |
| **MSE Loss** | **~0.2795** | Penalizes severe prediction outliers during training. |

---

## 🧠 Model Architecture

Built using Keras Sequential API with a funnel architecture, regularized with **Batch Normalization** and **Dropout**:
- Input (8 Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Lat, Long)
   -  ▼ Dense Layer (128 neurons, ReLU) Batch Normalization + Dropout (0.2) 
   -  ▼ Dense Layer (64 neurons, ReLU) Batch Normalization + Dropout (0.2) 
   -  ▼ Dense Layer (32 neurons, ReLU) 
   -  ▼ Output Layer (1 neuron, Linear)
