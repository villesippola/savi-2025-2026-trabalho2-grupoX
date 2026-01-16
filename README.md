<img width="1000" height="600" alt="training_curve_task4" src="https://github.com/user-attachments/assets/93eb69e5-f39d-42f3-a561-3700a0d17157" /># Practical Work 2 - SAVI
==============

Group 8

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

#### Layers of the model
<img width="1156" height="683" alt="Screenshot from 2026-01-15 21-16-45" src="https://github.com/user-attachments/assets/74538e37-3174-4bd0-91ab-e9c9239f5881" />


### Results

#### Training Curve

<img width="640" height="480" alt="training" src="https://github.com/user-attachments/assets/e8731878-76b8-476a-b051-cac6405b70d1" />

*Figure 1: Training and test loss over epochs. The best checkpoint is marked in green.*

The graph shows that the Train line does not fall below the Test line and the Test line does not rise above the Train line. `Dropout` layer successfully prevents overfitting.

#### Performance Metrics: improved model vs model from class

| Metric | ModelBetterCNN | ModelConvNet |
|--------|-------|-------|
| **Test Accuracy** | 0.9922 | 0.9795 |
| **Macro Precision** | 0.9921 | 0.9797 |
| **Macro Recall** | 0.9922 | 0.9790 |
| **Macro F1-Score** | 0.9921 | 0.9793 |
| **Total Params** | 1,701,578 | 421,642 |

Compared to the model developed in class (`ModelConvNet`), the new improved model (`ModelBetterCNN`) is much more accurate. However, the number of trainable parameters increases by about 4 times, making training much more time-consuming.

#### Confusion Matrix

<img width="640" height="480" alt="confusion_matrix" src="https://github.com/user-attachments/assets/eec9b919-ca89-4c82-9974-df6ec4f9b3e3" />

*Figure 2: Confusion Matrix showing the model's predictions vs ground truth.*

#### Per-Class Metrics from improved model

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

* **Total Images:** 60,000 (Train) / 10,000 (Test)
* **Variability:** We created 4 dataset versions to test robustness:
    * **Type A:** 1 Digit, Fixed Scale (28x28).
    * **Type B:** 1 Digit, Random Scale (22x22 to 36x36).
    * **Type C:** 3-5 Digits, Fixed Scale.
    * **Type D:** 3-5 Digits, Random Scale.

### Visualizations

<img width="2953" height="2984" alt="mosaic_vB" src="https://github.com/user-attachments/assets/a17b8f27-4669-4e94-855e-23ecdadee3a0" />

> *Figure 3: Mosaic of generated version B images with Ground Truth bounding boxes.*

<img width="2953" height="2984" alt="mosaic_vD" src="https://github.com/user-attachments/assets/b2e25455-19c6-42c0-9171-a04bb9c23010" />

> *Figure 4: Mosaic of generated version D images with Ground Truth bounding boxes.*

<img width="5968" height="3568" alt="statistics_vB" src="https://github.com/user-attachments/assets/741f0ec8-9498-4bc3-9581-f0f573b8d1f8" />

> *Figure 5: Distribution of classes, digits per image, and digit dimensions for version B.*

<img width="5992" height="3568" alt="statistics_vD" src="https://github.com/user-attachments/assets/4df9bb1e-63c8-4369-8fea-d717d7b2e6f1" />

> *Figure 6: Distribution of classes, digits per image, and digit dimensions for version D.* 

The graphs show that the size of the digits varies a lot and they also differ from the set range (22x22 to 36x36). This is because the program takes the dimensions from the tight boxes visible in the mosaics. In this case, for example, digit 1 has smaller dimensions than digit 5.

---

### Task 3: Sliding Window Detection

In this task, we implemented a **Sliding Window** approach to detect digits in the larger images generated in Task 2, utilizing the pre-trained CNN from Task 1 (`ModelBetterCNN`) as the backbone classifier.

**Methodology:**
* **Mechanism:** A window of size 28x28 pixels slides across the input image (128x128) with a step size (stride) of 6 pixels.
* **Inference:** Each cropped region is normalized and passed through the CNN.
* **Detection Criteria:** If the model's confidence for a predicted class exceeds a threshold of **0.98**, a bounding box is drawn around the window.

**Visual Results:**

We evaluated the performance on two dataset versions to illustrate the limitations of this approach.

**1. Results on Dataset vB (Single Digit):**
The green boxes represent detections with high confidence on images containing a single digit.

<img width="2250" height="2250" alt="mosaic_task3" src="https://github.com/user-attachments/assets/41217d84-1724-42e3-8671-c64f706857bf" />


> *Figure 7: Sliding Window detection on Dataset vB. The model successfully localizes the digit but generates False Positives in the background.*

**2. Results on Dataset vD (Multiple Digits):**
When applying the same method to complex scenes with multiple digits and variable scales, the issue becomes critical.

<img width="2250" height="2250" alt="mosaic_task3" src="https://github.com/user-attachments/assets/c402bbe3-24e7-46d9-bccd-3bee3ae06dd4" />

> *Figure 8: Sliding Window detection on Dataset vD. The accumulation of False Positives in the background creates a "noisy" detection, making it difficult to isolate the real digits cleanly.*

**Qualitative Evaluation:**
* **False Positives (The Background Problem):** As observed in Figures 7 and 8, the model frequently detects digits in empty black areas.
* **Root Cause:** The CNN was trained on the closed set of MNIST digits (0-9) and never encountered a "Background" or "Void" class during training. Consequently, the network forces empty space into one of the digit classes (often classifying noise or black pixels as a '1' or '7') with high confidence.
* **Performance:** This approach is computationally expensive as it requires thousands of forward passes per image and lacks precision in distinguishing objects from empty space. This limitation is addressed in Task 4.

---

### Task 4: Integrated Detector and Classifier

To overcome the limitations observed in Task 3 (specifically the high false positive rate in empty areas), we implemented an improved classification strategy by explicitly teaching the network to recognize the "background".

**Methodology:**

1. **Architecture Modification:**
   We updated the CNN architecture (`ModelTask4`) to output **11 classes** instead of the original 10.
   * Indices `0-9`: Represent the digits.
   * Index `10`: Represents the **Background/Void**.

2. **Retraining with Background Class:**
   We created a custom training loop (`train_task4.py`) that feeds the network with a mix of data:
   * **Positive Samples:** Standard MNIST images (labeled 0-9).
   * **Negative Samples:** Empty black images or noise generated on the fly (labeled as class 10).
   * This forces the network to learn features for "emptiness" rather than guessing a digit.

3. **Inference Logic:**
   During the Sliding Window process, we added a filter logic: any window classified as "Class 10" is immediately discarded as background, regardless of its confidence score.

---

**Quantitative Evaluation:**

Before visual testing, we validated the model's performance metrics to ensure it correctly learned to distinguish digits from the background.

**1. Training Stability:**
The training loss curve shows a rapid convergence, indicating that the model easily learned to separate the new "Background" class from the digits within 5 epochs.

<img width="1000" height="600" alt="training_curve_task4" src="https://github.com/user-attachments/assets/0fe3d7ff-af6a-422d-ab0e-5e60026c5c17" />

> *Figure 8: Training loss over epochs. The steep drop indicates effective learning.*

**2. Classification Metrics:**
The classification report confirms the robustness of the new model. Notably, the **"Fondo" (Background)** class achieved a **Precision and Recall of 1.00**, meaning the model almost never confuses the background with a digit.

<img width="585" height="415" alt="Classification_report" src="https://github.com/user-attachments/assets/3c84cc22-2da4-4554-9ccd-1ba518310260" />

> *Figure 9: Precision, Recall, and F1-Score per class. Note the perfect score for the background class.*

**3. Confusion Matrix:**
The confusion matrix visually demonstrates the separation between classes. The bottom-right square (Class "Fondo") is distinct, with zero confusion between the background and any digit.

<img width="1000" height="800" alt="confusion_matrix_task4" src="https://github.com/user-attachments/assets/b7aecb84-0884-4a01-8fe6-800ec4ddd69d" />

> *Figure 10: Confusion Matrix including the 11th class (Background). The diagonal shows high accuracy.*

---

**Visual Results (Sliding Window):**

After validating the metrics, we tested the detector on the test images.

**1. Results on Dataset vB (Single Digit):**
The network effectively filters out the black background. Unlike Task 3, there are no random green boxes in the empty space.

<img width="800" alt="mosaic_task4_vB" src="https://github.com/user-attachments/assets/fd27cbb6-cd8e-469e-9421-737186155243" />

> *Figure 11: Improved detection on Dataset vB. The false positives are completely eliminated.*

**2. Results on Dataset vD (Multiple Digits):**
The model is now capable of ignoring the background even in complex scenes with multiple objects. It successfully detects multiple digits without generating noise in the void areas.

<img width="2250" height="2250" alt="mosaic_task4" src="https://github.com/user-attachments/assets/a1e1127f-2aa5-4f04-a294-bd366211a265" />


> *Figure 12: Improved detection on Dataset vD. The detector isolates multiple digits cleanly.*

---

**Comparison (Task 3 vs Task 4):**

| Feature | Task 3 (Base Sliding Window) | Task 4 (Background Aware) |
| :--- | :--- | :--- |
| **Model Classes** | 10 (Digits only) | **11 (Digits + Background)** |
| **False Positives** | **High** (Detects digits in black space) | **Near Zero** |
| **Precision** | Low | **High** |
| **Visual Quality** | Cluttered with green boxes | Clean and focused |

**Conclusion:**
By explicitly modelling the "Background" class, we successfully transformed a simple classifier into a functional object detector capable of locating digits in a larger scene without being confused by empty space. This fulfills the requirement of creating a robust detector without needing complex architectures like YOLO for this specific dataset.
