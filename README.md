# 🏥 Elderly Care AI Monitoring System

A comprehensive AI-powered monitoring system for elderly care, built with Streamlit. This application provides real-time health monitoring, fall detection, cardiovascular risk assessment, expression analysis, and seniors' vital sign tracking, with integrated alert systems for immediate notifications.

---

## 📋 Table of Contents

- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Usage](#-usage)
- [Deployment](#-deployment)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## ✨ Features

### Core Modules

- 🏠 **Dashboard Overview**: Real-time metrics, alert logs, and system status.
- ❤️ **Cardio Risk Checker**: Assess cardiovascular risk based on user inputs (age, height, weight, BP, cholesterol).
- 😊 **Expression Detection**: Analyze facial expressions for emotional health monitoring.
- 👴 **Seniors Monitoring**: Track vital signs (body temperature, heart rate, SPO2) and predict health status.
- 🧍‍♂️ **Fall Detection**: Upload images to detect falls, standing, or sitting using a trained ResNet50 model.

### Notification & Alert Systems

- **Push Notifications**: Send alerts via Firebase Cloud Messaging (FCM) to mobile devices.
- **Buzzer Alerts**: Simulate or integrate hardware buzzers for on-site alerts.
- **SMS Notifications**: Send alerts via Twilio (configurable).
- **Email Notifications**: SMTP-based email alerts.
- **Configurable Alerts**: Enable/disable notifications via sidebar settings.

### Additional Features

- Interactive visualizations (charts, metrics, dataframes).
- Real-time predictions with confidence scores.
- Model performance metrics and confusion matrices.
- Responsive UI with sidebar navigation.
- Test alert system for validation.

---

## 🛠 Technologies Used

- **Frontend/UI**: Streamlit
- **AI/ML**: PyTorch, Scikit-learn, Joblib
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Notifications**:
  - Firebase (Push)
  - Twilio (SMS)
  - SMTP (Email)
  - GPIO/Raspberry Pi (Buzzer, optional)
- **Other**: Requests, JSON, PIL (for image handling)

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- Git
- (Optional) Raspberry Pi or Arduino for hardware buzzer integration

### Steps

**Clone the Repository:**
git clone https://github.com/your-username/elderly-care-monitoring.git
cd elderly-care-monitoring

**Set Up Models and Data:**
- Place trained models (e.g., `fall_detection_model.pth`, `cardio_model.pkl`) in the `models/` folder.
- Place datasets (e.g., CSVs) in the `datasets/` folder.
- Update file paths in the code if necessary.

**Configure APIs (Optional):**
- For push notifications: Get FCM server key from [Firebase Console](https://console.firebase.google.com/).
- For SMS: Sign up at [Twilio](https://www.twilio.com/) and get credentials.
- For email: Set up Gmail app password.
- Update the placeholders in `app.py` or use Streamlit secrets for security.

**Run Locally:**
Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🚀 Usage

### Navigate the App

- Use the sidebar to select modules (Dashboard, Cardio Risk, etc.).
- Configure alerts in the sidebar (enable push, SMS, etc.).

### Dashboard

- View metrics and recent alerts.
- Test the alert system with the "Test Alert" button.

### Predictions

- **Cardio Risk**: Input health details and get risk probability.
- **Expression Detection**: Upload images for analysis.
- **Seniors Monitoring**: Enter vitals and predict status.
- **Fall Detection**: Upload images to detect falls with visualizations.

### Alerts

- Alerts trigger automatically on critical predictions (e.g., high cardio risk or fall detected).
- Customize recipients in the sidebar.

---

## 🌐 Deployment

### Streamlit Cloud (Recommended)

1. Push code to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and connect your repo.
3. Select `app.py` as the main file and deploy.
4. Add secrets (API keys) via the Streamlit Cloud dashboard.

### Other Options

- **Heroku**: Follow [Streamlit's Heroku guide](https://devcenter.heroku.com/articles/getting-started-with-python).
- **Local with Ngrok**: Run locally and use `ngrok http 8501` for temporary public access.
- **AWS/Azure**: Deploy on cloud VMs with Docker.

---

## 📁 Project Structure
elderly-care-monitoring/
├── app.py # Main Streamlit app
├── requirements.txt # Dependencies
├── pages/ # Sub-modules
│ ├── cardio.py
│ ├── expression.py
│ ├── seniors.py
│ └── fall.py
├── models/ # Trained models
│ ├── fall_detection_model.pth
│ ├── cardio_model.pkl
│ └── ...
├── datasets/ # Data files (CSVs, etc.)
│ └── ...
└── README.md # This file

---

## ⚙ Configuration

- **File Paths**: Update paths in `app.py` and page files for your environment.
- **Secrets**: Use `st.secrets` for API keys instead of hardcoding.
- **Hardware**: For buzzer, uncomment GPIO code and set `BUZZER_PIN`.
- **Models**: Ensure models are compatible (e.g., ResNet50 for fall detection).

---

## 🤝 Contributing

- Fork the repository.
- Create a feature branch (`git checkout -b feature/new-feature`).
- Commit changes (`git commit -am 'Add new feature'`).
- Push to the branch (`git push origin feature/new-feature`).
- Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 📞 Contact

**Author:** Shivangi Sahu  
**Email:** shivi8f67@gmail.com  
**GitHub:** https://github.com/Er-shivangisahu2030
(#)

---


