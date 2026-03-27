# 🚦 ASTRA: Advanced Smart Traffic AI Management System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

**ASTRA** is an end-to-end, real-time AI traffic optimization platform. Designed to solve urban congestion and improve emergency response times, ASTRA fuses Computer Vision, Reinforcement Learning, and Multi-Modal AI into a single, highly optimized edge-computing application.

---

## ✨ Core System Features

* **👁️ Real-Time Computer Vision:** Processes live video feeds to detect, classify, and track specific vehicle types (Cars, Trucks, Buses, Motorcycles) and Pedestrians to calculate precise lane density.
* **🧠 Q-Learning Optimization (RL):** Replaces static timers with a dynamic Reinforcement Learning algorithm that continuously adjusts green-light durations based on live traffic states and a custom reward function.
* **🚑 Acoustic Emergency Preemption:** Utilizes simulated audio detection to identify approaching ambulance sirens, instantly preempting the normal traffic cycle to clear intersections.
* **🌫️ Environmental Adaptation:** Analyzes the Laplacian Variance of video frames to detect fog, rain, or low-light conditions. Automatically lowers AI confidence thresholds and applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to restore visibility.
* **🚨 Live Telemetry & Alerts:** Integrated with the Telegram Bot API to push synchronous text alerts for critical congestion events and Red-Light Violations (including vehicle tracking IDs).
* **📊 Predictive Analytics Dashboard:** A multi-page SQLite-backed dashboard providing historical traffic flow analysis, peak-hour prediction (via Linear Regression), and CSV data exports.

---

## 🏗️ The Tech Stack & Architecture

### Why YOLOv5? 
In traffic management, **latency is lethal**. While models like Faster R-CNN offer incredible accuracy, they are too heavy for real-time edge processing without massive GPUs. **YOLOv5** was specifically chosen because it hits the perfect mathematical sweet spot for Edge AI:
1.  **Hardware Efficiency:** It delivers ~30 FPS on standard CPUs and entry-level GPUs, making it viable for deployment on standard street-light hardware.
2.  **Stability & Exportability:** The PyTorch architecture of YOLOv5 is exceptionally stable and easily integrates directly with OpenCV without heavy API overhead.

### Why Streamlit? 
Building a standard web app (React frontend + Django/FastAPI backend) requires pushing massive video tensors over WebSockets, causing severe latency and memory bottlenecks. **Streamlit** was selected because:
1.  **Zero-Latency Tensor Rendering:** Streamlit allows us to process PyTorch tensors and OpenCV NumPy arrays in backend memory and render them directly to the frontend without serialization delays.
2.  **Rapid Dashboarding:** It natively handles Matplotlib charts and Pandas DataFrames, allowing the AI logic and the Data Analytics dashboard to live seamlessly in the same centralized codebase.

### Additional Technologies:
* **OpenCV (`cv2`):** Used for matrix transformations, custom Centroid Tracking algorithms, and high-speed hardware scanning graphics.
* **Scikit-Learn:** Powers the Linear Regression model for predicting future traffic intervals.
* **SQLite3:** A lightweight, serverless database ideal for local IoT logging without the overhead of PostgreSQL.

---

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.8 or higher
* A webcam, or a mobile camera connected via DroidCam/IP Webcam.

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/Astra-Traffic-AI.git](https://github.com/yourusername/Astra-Traffic-AI.git)
cd Astra-Traffic-AI
