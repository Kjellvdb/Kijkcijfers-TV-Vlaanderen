# 📊 TV Viewer Forecasting System

This project develops a full pipeline to predict the number of viewers for TV programs using rich time series and tabular data. It covers preprocessing, feature engineering, modeling, and tuning to achieve high accuracy in viewer predictions.

## ✅ Requirements

- Python ≥ 3.11
- All required dependencies are listed in [requirements.txt](requirements.txt)

Key libraries used:

- ML: `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
- Data: `pandas`, `numpy`, `holidays`, `category_encoders`
- Visualization: `matplotlib`, `seaborn`
- Utilities: `joblib`, `openmeteo_requests`, `requests_cache`, `retry_requests`

## 📦 Setup Instructions

1. **Clone the repository**

   ```bash
   git clone git@github.com:Kjellvdb/Kijkcijfers-TV-Vlaanderen.git
   cd Kijkcijfers-TV-Vlaanderen
   ```

2. **Setup virtual environment**

   Linux/MacOS:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   Windows:

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

To make predictions on new program data provided in `data.csv`:

```bash
python main.py
```

This will:

- Load your pre-trained model from model.pkl
- Preprocess and validate data.csv
- Output viewer predictions to `predictions.csv`

### 📌 Note

Ensure `data.csv` is formatted correctly with the required columns:

| Column    | Description               | Format       |
| :-------- | :------------------------ | :----------- |
| Programma | Title of the TV program   | String       |
| Zender    | Channel name              | String       |
| Datum     | Date of the program       | `dd/mm/yyyy` |
| Start     | Start time of the program | `HH:MM:SS`   |
| Duur      | Duration of the program   | `HH:MM:SS`   |

## 📁 Projectstructure

```plaintext
.
├── data/                    # Raw and processed datasets
│   ├── cat.csv              # Program categories
│   ├── cim.csv              # Program metadata
│   ├── kijkcijfers.csv      # Processed data
│   └── weather.csv          # Weather data
├── scripts/                 # Jupyter notebooks for data retrieval and modeling
│   ├── kijkcijfers.ipynb    # Main modeling and analysis notebook
│   └── retrieveData.ipynb   # External data fetching (program data, weather data)
├── util/                    # Helper resources
│   ├── categories.csv       # List of valid categories
│   ├── channels.csv         # List of valid channels
│   ├── primeChannels.csv    # List of prime channels
│   ├── lastOccurrences.csv  # Historical title occurrences
│   └── timeSeriesData.csv   # Historical viewer data
├── main.py                  # Main script to generate predictions from `data.csv`
├── data.csv                 # New input data for prediction
├── predictions.csv          # Output predictions from the model
├── model.pkl                # Trained model
├── requirements.txt         # Python dependency list
├── .gitignore               # Files and directories to ignore in git
├── .gitattributes           # Git attributes for repository
└── README.md                # You're reading this
```

## ⚙️ Features

- 📅 Temporal feature engineering, including holiday recognition using `holidays`
- 📐 Automated extraction of categorical and numerical features from metadata
- 🔎 Seasonal decomposition for time series features
- 🧠 Support for multiple regression models:
  - Linear Regression, Ridge, Random Forest, Extra Trees
  - Gradient Boosting, XGBoost, LightGBM, CatBoost
  - Ensemble models: Voting, Bagging, Stacking
- 📈 Evaluation using RMSE and MAPE
- 🧪 Hyperparameter optimization via `GridSearchCV`
- 💾 Model persistence with `joblib`
