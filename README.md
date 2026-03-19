# 🌾 Crop Recommendation System

A smart crop recommendation tool that uses machine learning to suggest the most suitable crop based on your soil and weather conditions. Just enter a few details about your field — like nitrogen, phosphorus, potassium levels, temperature, humidity, pH, and rainfall — and the model tells you what to grow.

## How It Works

The system is built around a **Random Forest Classifier** trained on a dataset of 2,200 samples covering 22 different crops. Under the hood there are two main pieces:

- **Flask API** — a lightweight backend that loads the trained model and serves predictions over a simple REST endpoint.
- **Streamlit UI** — a clean, interactive frontend where you can adjust sliders for each input parameter and get an instant recommendation.

## Project Structure

```
Crop-Reccomendation-System/
├── train_model.py            # Trains and saves the ML model
├── backend/
│   ├── app.py                # Flask REST API
│   └── utils.py              # Input validation helpers
├── frontend/
│   └── streamlit_app.py      # Streamlit web UI
├── Crop_recommendation.csv   # Dataset (2,200 samples, 22 crops)
├── model.pkl                 # Pre-trained model
├── correlation_heatmap.png   # Feature correlation heatmap
├── feature_importance.png    # Feature importance chart
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/KaranP315/Crop-Reccomendation-System.git
cd Crop-Reccomendation-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model (optional — a pre-trained `model.pkl` is included)

```bash
python train_model.py
```

This loads the dataset, trains the Random Forest model, prints the accuracy report, and saves `model.pkl` along with a couple of visualisation charts.

### 4. Start the backend

```bash
python backend/app.py
```

The Flask server starts at **http://127.0.0.1:5000**.

You can also test it directly with curl:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"N\":90,\"P\":42,\"K\":43,\"temperature\":20.87,\"humidity\":82,\"ph\":6.5,\"rainfall\":202.9}"
```

Expected response:

```json
{ "recommended_crop": "rice" }
```

### 5. Launch the frontend

```bash
streamlit run frontend/streamlit_app.py
```

Open the URL shown in the terminal (usually **http://localhost:8501**). Make sure the Flask backend is already running before you use the UI.

## Dataset Features

| Feature     | What it measures            |
|-------------|-----------------------------|
| Nitrogen    | N content in the soil       |
| Phosphorus  | P content in the soil       |
| Potassium   | K content in the soil       |
| Temperature | Temperature in °C           |
| Humidity    | Relative humidity (%)       |
| pH          | Soil pH value               |
| Rainfall    | Rainfall in mm              |

The model predicts one of **22 crop labels** based on these seven inputs.

## Tech Stack

- **Machine Learning** — pandas, NumPy, scikit-learn
- **Backend** — Flask, Flask-CORS
- **Frontend** — Streamlit
- **Visualisation** — matplotlib, seaborn

## License

This project is intended for educational and learning purposes.
