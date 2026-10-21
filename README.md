# 🌿 VERDANT — Crop Yield Intelligence

### Machine Learning-Based Crop Yield Prediction Web Application

VERDANT is a web-based crop yield prediction application designed to help users estimate agricultural productivity using machine learning. Built with **Python, Scikit-learn, Pandas, Joblib, and Streamlit**, the application provides an interactive dashboard where users can enter crop and environmental information to predict expected yield.

The application combines a modern agricultural dashboard with a trained Random Forest regression model to support data-driven farming decisions.

## ✨ Features

- 🌱 **Crop Yield Prediction** — Estimate crop yield in tons per hectare.
- 🌍 **Farm Information Inputs** — Enter region, crop, soil type, and weather conditions.
- 🌡️ **Environmental Factors** — Provide temperature, rainfall, and days to harvest.
- 💧 **Farming Practice Inputs** — Specify fertilizer and irrigation usage.
- 📊 **Interactive Dashboard** — View predictions, session metrics, and yield charts.
- 💬 **Built-in Farming Assistant** — Get predefined answers to common questions about the application and its predictions.
- 📥 **Downloadable Reports** — Export individual prediction reports and prediction history as CSV.
- 🌿 **Modern UI** — Forest-green sidebar, sage-inspired styling, cream backgrounds, and responsive dashboard cards.
- 🤖 **Machine Learning Integration** — Uses a saved Scikit-learn model to generate predictions.

## 🖥️ Dashboard

The VERDANT dashboard includes:

- Home dashboard
- Predict Yield
- Farming Assistant
- Insights
- About the Model

The interface is designed around a clean agricultural theme to make the prediction workflow intuitive and accessible.

## 🧠 Machine Learning Model

The application uses a trained **Random Forest Regressor** with a preprocessing pipeline that handles categorical and numerical features.

### Input Features

The model expects the following nine features:

| Feature | Description |
|---|---|
| `Region` | Agricultural region |
| `Weather_Condition` | Weather condition |
| `Crop` | Crop type |
| `Soil_Type` | Soil category |
| `Temperature_Celsius` | Temperature in Celsius |
| `Rainfall_mm` | Rainfall in millimetres |
| `Fertilizer_Used` | Whether fertilizer was used |
| `Irrigation_Used` | Whether irrigation was used |
| `Days_to_Harvest` | Number of days to harvest |

### Model Performance

Recorded evaluation results from the trained model:

| Metric | Value |
|---|---:|
| R² Score | 0.9129 (91.29%) |
| Mean Absolute Error (MAE) | 0.3990 |
| Root Mean Squared Error (RMSE) | 0.5001 |

These values describe the model's earlier evaluation results. Performance on new data may differ.

## 🛠️ Technologies Used

- **Python** — Application and model integration
- **Streamlit** — Interactive web application
- **Pandas** — Data handling and prediction inputs
- **NumPy** — Numerical computing
- **Scikit-learn** — Machine learning and preprocessing
- **Joblib** — Saving and loading the trained model
- **HTML and CSS** — Dashboard styling

## 📁 Project Structure

```text
Crop App/
│
├── app.py
├── crop_yield_prediction_model.pkl
├── requirements.txt
├── Crop_Yield_Prediction(2).ipynb
└── README.md
```

- `app.py` — Streamlit dashboard and application logic.
- `crop_yield_prediction_model.pkl` — Saved trained model and preprocessing pipeline.
- `requirements.txt` — Python dependencies.
- `Crop_Yield_Prediction(2).ipynb` — Notebook used for data analysis and model development, if included in the repository.
- `README.md` — Project documentation.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/crop-yield-prediction.git
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Navigate to the project directory

```bash
cd crop-yield-prediction
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser to use the application.

## 📦 Requirements

Create a `requirements.txt` file with the following packages:

```text
streamlit
pandas
numpy
scikit-learn
joblib
```

For reproducible deployment, consider pinning the versions used to train and run the model.

## ☁️ Deploy on Streamlit Community Cloud

1. Push the project files to a GitHub repository.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in using your GitHub account.
4. Select **Create app** and choose your repository.
5. Select the correct branch and set the main file path to `app.py`.
6. Click **Deploy**.

Ensure the model file is available to the deployed application and that the required dependencies are listed in `requirements.txt`.

## 📈 How to Use

1. Launch the VERDANT application.
2. Open **Home** or **Predict Yield**.
3. Enter the nine required crop and environmental features.
4. Click **Predict Yield**.
5. Review the predicted yield in tons per hectare.
6. Download the prediction report if required.
7. Visit **Insights** to review predictions made during the current session.
8. Use **Farming Assistant** for answers to common application questions.

## ⚠️ Limitations

- Predictions are estimates, not guarantees of actual harvest outcomes.
- Accuracy depends on the training dataset and how closely new inputs match the data used during training.
- The farming assistant uses predefined responses and is not a live AI chatbot.
- Prediction history is stored in the current Streamlit session and may not persist after the session ends.
- The application does not replace professional agricultural advice or local soil testing.

## 🔮 Future Improvements

- Integrate live weather data.
- Add crop-specific recommendations.
- Store prediction history in a database.
- Add downloadable PDF reports.
- Improve the farming assistant with a language model.
- Add visual analytics for crop and regional trends.
- Incorporate additional agricultural datasets and external validation.

## 🎯 Project Goal

The goal of VERDANT is to demonstrate how machine learning and interactive web technologies can support agricultural data analysis and help users explore crop yield estimates.

## 👩‍💻 Author

**Your Name**

GitHub: [YOUR-USERNAME](https://github.com/YOUR-USERNAME)

## 📄 License

This project is available for educational and research purposes. Add a `LICENSE` file to the repository if you want to specify the terms under which others may use, modify, and distribute the project.

---

**VERDANT — Grow Smarter. Harvest Better. 🌿**
