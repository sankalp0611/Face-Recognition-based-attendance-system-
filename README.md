# Face-Recognition-based-attendance-system- # Face Recognition-Based Attendance Management System  

## Overview  
This project implements a **face recognition-based attendance management system** that automates attendance tracking in educational institutions. By leveraging **FaceNet embeddings and an SVM classifier**, the system ensures **efficient, accurate, and secure** attendance recording.  

## Features  
- **Automated Face Recognition:** Uses FaceNet for feature extraction and SVM for classification.  
- **Real-Time Attendance Marking:** Supports both **image and video inputs** for flexible recognition.  
- **Cross-Platform Accessibility:** Web and mobile interfaces (Android/iOS) for easy usage.  
- **Secure Data Storage:** AES-256 encryption ensures data integrity and privacy.  
- **Scalability:** Designed to handle large datasets efficiently.  
- **Performance Optimization:** High classification accuracy with robust feature extraction.  

## Dataset  
- **Total Students:** 93  
- **Images per Student:** ~20  
- **Total Images:** 1860  
- **Preprocessing:**  
  - Face detection using **MTCNN**  
  - Cropped to **160 × 160 pixels**  
- **Feature Extraction:** 128-dimensional embeddings using **FaceNet**  
- **Classification Model:** **SVM** trained with an **80:20 train-test split**  

## Performance Evaluation  
- **Accuracy:** High recognition accuracy with robust generalization.  
- **Metrics:** Precision, Recall, and F1-score showed consistent performance.  
- **Visualization:** Confusion matrix and t-SNE plots confirmed distinct feature embedding for each student.  

## Installation  

### Prerequisites  
Ensure you have the following installed:  
- Python 3.x  
- TensorFlow  
- OpenCV  
- Scikit-Learn  
- Flask/Django (for web application)  
- React Native (for mobile interface)  

### Clone Repository  
```sh
git clone https://github.com/your-username/face-recognition-attendance.git
cd face-recognition-attendance
