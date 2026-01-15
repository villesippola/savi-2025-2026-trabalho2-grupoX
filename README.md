# Practical Work 2 - SAVI
==============

Group X

2025-2026

## Tasks

---

### Task 1: Optimized CNN Classifier (Full MNIST)

To classify individual MNIST digits, we developed a custom Convolutional Neural Network (CNN) using PyTorch.

* **Architecture:** The model (`ModelBetterCNN`) improves upon the baseline by increasing network depth and adding regularization mechanisms.
    * **Input:** 28x28 Grayscale images.
    * **Layers:** We utilized 3 Convolutional blocks. Each block consists of:
        * `Conv2d`: Feature extraction.
        * `BatchNorm2d`: To stabilize training and allow higher learning rates.
        * `ReLU`: Non-linear activation.
        * `MaxPool2d`: To reduce spatial dimensions and computation.
    * **Regularization:** `Dropout` layers were added before the fully connected layers to prevent overfitting.

### Layers of the model
<img width="1156" height="683" alt="Screenshot from 2026-01-15 21-16-45" src="https://github.com/user-attachments/assets/74538e37-3174-4bd0-91ab-e9c9239f5881" />


### Results

#### Training Curve

<img width="640" height="480" alt="training" src="https://github.com/user-attachments/assets/e8731878-76b8-476a-b051-cac6405b70d1" />

*Figure 1: Training and test loss over epochs. The best checkpoint is marked in green.*

#### Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 0.9922 |
| **Macro Precision** | 0.9921 |
| **Macro Recall** | 0.9922 |
| **Macro F1-Score** | 0.9921 |

#### Confusion Matrix

<img width="640" height="480" alt="confusion_matrix" src="https://github.com/user-attachments/assets/eec9b919-ca89-4c82-9974-df6ec4f9b3e3" />

*Figure 2: Confusion Matrix showing the model's predictions vs ground truth.*

#### Per-Class Metrics

| Digit | Precision | Recall | F1-Score |
| --- | --- | --- | --- |
| 0 | 0.9910 | 0.9980 | 0.9944 |
| 1 | 0.9947 | 0.9965 | 0.9956 |
| 2 | 0.9942 | 0.9913 | 0.9927 |
| 3 | 0.9950 | 0.9941 | 0.9946 |
| 4 | 0.9869 | 0.9939 | 0.9904 |
| 5 | 0.9922 | 0.9922 | 0.9922 |
| 6 | 0.9906 | 0.9916 | 0.9911 |
| 7 | 0.9913 | 0.9951 | 0.9932 |
| 8 | 0.9918 | 0.9938 | 0.9928 |
| 9 | 0.9939 | 0.9752 | 0.9845 |

---

### Task 2: Synthetic Dataset Generation

We generated a "Scene" dataset to simulate object detection tasks. The generation script (`generate_data.py`) places MNIST digits onto a larger canvas (128x128) while preventing overlap.

* **Variability:** We created 4 dataset versions to test robustness:
    * **Type A:** 1 Digit, Fixed Scale (28x28).
    * **Type B:** 1 Digit, Random Scale (22x22 to 36x36).
    * **Type C:** 3-5 Digits, Fixed Scale.
    * **Type D:** 3-5 Digits, Random Scale.

---

### Task 3: Sliding Window Detection



---

### Task 4: Integrated Detector and Classifier



---
