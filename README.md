#  Crop Recommendation System

Choosing the right crop for your field can be tricky — it depends on soil quality, weather patterns, and dozens of other factors that are hard to evaluate by hand. This project takes the guesswork out of it.

The **Crop Recommendation System** is a machine learning application that analyzes your soil and climate conditions and recommends the best crop to grow. You provide seven simple inputs — nitrogen, phosphorus, potassium levels, temperature, humidity, pH, and rainfall — and the model does the rest. It's designed to be a practical tool for farmers, agriculture students, and anyone curious about data-driven farming.

## Why This Project?

Traditional farming decisions are often based on experience or guesswork. But with soil testing becoming more accessible, there's an opportunity to use that data more effectively. This project demonstrates how a straightforward ML model can turn those numbers into an actionable recommendation — no deep learning or complex infrastructure required.

## What It Can Do

- **Predict the best crop** from 22 options (rice, wheat, maize, cotton, coffee, mango, banana, and more) based on real soil and weather parameters
- **Validate inputs** to make sure values are realistic before making a prediction
- **Serve predictions via API**, so it can be integrated into other tools or apps if needed
- **Provide an interactive web interface** where you can tweak sliders and get instant results without writing any code

## How It Works Under the Hood

The core of the system is a **Random Forest Classifier** — an ensemble learning method that builds 100 decision trees and aggregates their predictions. It was trained on a publicly available dataset of **2,200 samples** covering **22 crop types**, with each sample containing 7 measured features.

The trained model achieves roughly **99% accuracy** on the test set, which makes sense given the dataset — the crop classes are well-separated in the feature space, meaning each crop has fairly distinct soil and weather requirements.

The application is split into three parts:

1. **Training Script** (`train_model.py`) — Loads the dataset, trains the Random Forest model, evaluates it, and saves the trained model as a `.pkl` file. Also generates a correlation heatmap and feature importance chart for analysis.

2. **Backend API** (`backend/app.py`) — A Flask server that loads the trained model at startup and exposes a `/predict` endpoint. You send it a JSON payload with the 7 input features, and it returns the recommended crop. Input validation is handled by a separate utility module (`backend/utils.py`) to keep things clean.

3. **Frontend UI** (`frontend/streamlit_app.py`) — A Streamlit web app with sliders for each input parameter. Hit the predict button, and it calls the Flask API behind the scenes and displays the result in a styled card.

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

### 3. Train the model (optional — a pre-trained `model.pkl` is already included)

```bash
python train_model.py
```

This loads the dataset, trains the Random Forest model, prints the accuracy and classification report, and saves `model.pkl` along with two visualisation charts.

### 4. Start the backend

```bash
python backend/app.py
```

The Flask server starts at **http://127.0.0.1:5000**.

You can test it directly with curl:

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

## Dataset

The model is trained on the [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) which contains 2,200 rows and 8 columns:

| Feature     | What it measures              | Range        |
|-------------|-------------------------------|--------------|
| Nitrogen    | N content in the soil         | 0 – 140      |
| Phosphorus  | P content in the soil         | 5 – 145      |
| Potassium   | K content in the soil         | 5 – 205      |
| Temperature | Temperature in °C             | 0 – 50       |
| Humidity    | Relative humidity (%)         | 10 – 100     |
| pH          | Soil pH value                 | 0 – 14       |
| Rainfall    | Rainfall in mm                | 20 – 300     |

The target column contains one of **22 crop labels**: rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, and coffee.

## Tech Stack

| Layer              | Technology                   |
|--------------------|------------------------------|
| Machine Learning   | pandas, NumPy, scikit-learn  |
| Backend            | Flask, Flask-CORS            |
| Frontend           | Streamlit                    |
| Visualisation      | matplotlib, seaborn          |
| Model Serialisation| joblib                       |

## License

This project is intended for educational and learning purposes.
