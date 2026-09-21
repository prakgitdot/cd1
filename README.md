# AI-Based Crop Recommendation System

A Flask web app that recommends the most suitable crop for a plot of
land based on seven inputs: Nitrogen (N), Phosphorus (P), Potassium (K),
temperature, humidity, soil pH, and rainfall.

## Project structure

```
crop-recommendation-app/
├── app.py                    # Flask entry point (production-ready)
├── requirements.txt          # Pinned Python dependencies
├── Procfile                  # Cloud start command (gunicorn)
├── runtime.txt               # Python version for the cloud platform
├── .gitignore
├── data/
│   ├── generate_dataset.py   # Generates the training dataset
│   └── Crop_recommendation.csv
├── model/
│   ├── train_model.py        # Trains and saves the model
│   ├── crop_model.pkl        # Trained RandomForestClassifier
│   ├── scaler.pkl            # Fitted StandardScaler
│   └── label_encoder.pkl     # Fitted LabelEncoder (crop name <-> index)
├── templates/
│   └── index.html            # Frontend page
└── static/
    ├── style.css
    └── script.js
```

## Tech stack

- **Backend:** Flask 3, served in production by gunicorn
- **ML model:** scikit-learn `RandomForestClassifier` (22 crop classes)
- **Frontend:** plain HTML/CSS/JS (no build step required)

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Visit http://localhost:5000

## Retraining the model (optional)

The trained model is already included in `model/`. You only need to
retrain if you change the dataset or want to try a different algorithm:

```bash
python data/generate_dataset.py   # regenerate the dataset (optional)
python model/train_model.py       # retrain and overwrite model/*.pkl
```

## API

`POST /predict` — JSON body:

```json
{"N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82, "ph": 6.5, "rainfall": 202}
```

Response:

```json
{"success": true, "recommended_crop": "rice", "top_3": [...]}
```

`GET /health` — returns `{"status": "ok"}` if the model loaded correctly.

## Deployment

See the deployment guide provided alongside this project for step-by-step
GitHub + Render (or similar) cloud deployment instructions.
