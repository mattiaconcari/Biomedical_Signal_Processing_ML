# Biomedical Signal Processing & ML: Human Activity & Respiratory Monitoring 🏥📊

## 📌 Project Overview
This repository illustrates a dual-case study in biomedical signal processing and Machine Learning. The objective was to acquire raw physiological data from different sensor technologies, process the signals, and train a classification model to identify specific states.

### Case Study 1: Human Activity Recognition (HAR) via Wearables
* **Objective:** Classifying human motion states (Walking, Running, Standing) based on inertial data.
* **Workflow:** Data acquisition via smartphone IMU sensors (accelerometers/gyroscopes), real-time filtering in **LabVIEW**, and multi-class classification using a **K-Nearest Neighbors (KNN)** algorithm in **MATLAB**.
* 📁 **[View Code & Files](./Human_Activity_Recognition/)**
* 📄 **[Read the full Activity Recognition Report (PDF)](./Human_Activity_Recognition/Activity_Recognition_Report.pdf)**

### Case Study 2: Respiratory Monitoring via FBG
* **Objective:** Extracting and classifying respiratory patterns into three distinct physiological categories.
* **Workflow:** Processing of optical signals from **Fiber Bragg Grating (FBG)** sensors to isolate respiratory cycles from motion artifacts. The extracted features were then fed into the **KNN** classifier in **MATLAB** to automatically categorize the breathing types.
* 📁 **[View Code & Files](./Respiratory_FBG/)**
* 📄 **[Read the full Respiratory FBG Report (PDF)](./Respiratory_FBG/Respiratory_FBG_Report.pdf)**

---

## 🛠️ Tech Stack & Tools
* **Acquisition & Signal Processing:** LabVIEW
* **Machine Learning:** MATLAB (Statistics and Machine Learning Toolbox)
* **Algorithms:** K-Nearest Neighbors (KNN) for multi-class categorization
* **Sensors:** Smartphone IMU (Kinematic), FBG (Optical/Strain)
