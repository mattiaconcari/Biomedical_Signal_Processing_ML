# Biomedical Signal Processing & ML: Gait & Respiratory Monitoring 🏥📊

![Signal Processing](images/signal_plot.png)
*Caption: Example of filtered signal and feature extraction for activity classification.*

## 📌 Project Overview
This repository illustrates a dual-case study in biomedical signal processing, focusing on human activity classification and physiological monitoring through different sensor technologies.

### Case Study 1: Wearable Gait Analysis
* **Objective:** Classifying human motion (Walking, Running, Standing).
* **Workflow:** Data acquisition via smartphone IMU sensors, real-time filtering and feature extraction in **LabVIEW**, and final classification using a **K-Nearest Neighbors (KNN)** algorithm in **MATLAB**.

### Case Study 2: Respiratory Monitoring via FBG
* **Objective:** Extracting respiratory rate from high-sensitivity optical sensors.
* **Workflow:** Processing of optical signals from **Fiber Bragg Grating (FBG)** sensors to isolate respiratory patterns from motion artifacts and noise. The extracted features were then fed into the **KNN** classifier to automatically categorize the breathing types.

---

## 🛠️ Tech Stack & Tools
* **Acquisition & Filtering:** LabVIEW
* **Machine Learning:** MATLAB (Statistics and Machine Learning Toolbox)
* **Algorithms:** K-Nearest Neighbors (KNN)
* **Sensors:** Smartphone IMU (Accelerometer/Gyroscope), FBG (Optical)
