# BMI Prediction App

## Setup

```bash
pip install flask flask-cors tensorflow pillow
```

## Folder structure
```
bmi_app/
├── app.py
├── static/
│   └── index.html
└── models/
    ├── vgg16_bmi.keras
    ├── vggface_bmi.keras
    └── vggface_gender_bmi.keras
```

## Run
1. Download the 3 `.keras` model files from the shared Drive into `models/`
2. `python app.py`
3. Open `http://localhost:5000`
