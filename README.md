# Medi Assistant AI 🏥

### AI-Powered Medical Symptom Checker and Healthcare Assistance System

Medi Assistant AI is a machine learning-based healthcare assistance project that analyzes user-selected symptoms and predicts possible health conditions. It provides educational medicine information and healthcare guidance through an interactive Gradio interface.

## ✨ Features

- **AI Disease Prediction:** Predicts possible health conditions based on selected symptoms.
- **Medicine Information:** Provides educational information related to predicted conditions.
- **Health Risk Assessment:** Identifies potential health risks using predefined rules.
- **Doctor Recommendation:** Suggests relevant medical specialists.
- **Disease Information:** Provides basic information about different health conditions.
- **Interactive Interface:** User-friendly interface developed using Gradio.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Gradio
- Jupyter Notebook

## 🤖 Machine Learning Models

The project explores multiple classification algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

Logistic Regression achieved approximately **76% test accuracy** in the current experiment.

## 📁 Project Structure

```text
Medi-Assistant-AI/
│
├── dataset/
│   └── medical_symptom_checker_synthetic_500.csv
│
├── models/
│   ├── best_model.pkl
│   ├── model_metadata.json
│   └── model_metadata.pkl
│
├── notebooks/
│   └── Medi_Assistant_AI.ipynb
│
├── requirements.txt
└── README.md
```

## ⚙️ Installation

1. Clone this repository:

```bash
git clone https://github.com/YOUR-USERNAME/Medi-Assistant-AI.git
```

2. Navigate to the project directory:

```bash
cd Medi-Assistant-AI
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Open the Jupyter Notebook:

```bash
jupyter notebook
```

5. Open `notebooks/Medi_Assistant_AI.ipynb` and run the cells.

## 📊 Model Performance

| Algorithm | Test Accuracy |
|---|---:|
| Logistic Regression | 76% |
| Random Forest | 74% |
| Support Vector Machine | 74% |
| Decision Tree | 54% |

*Results are from the current experimental dataset and may vary with different data splits.*

## 🔮 Future Scope

- Integration with larger, clinically validated datasets.
- Improved machine learning prediction performance.
- Integration of verified medical information sources.
- Development of a web and mobile application.
- Clinical evaluation with qualified healthcare professionals.

  
⭐ If you find this project interesting, consider giving this repository a star!
