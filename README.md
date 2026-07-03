# 🎯 Best-Fit Career Field Predictor

An end-to-end machine learning system that recommends the most suitable
academic/career field for an engineering student — **AI-ML, Web Development,
Core CS, Data Science, or Cybersecurity** — based on academic performance,
self-rated skills, and project/activity history.

Built as a tabular multi-class classification problem using **XGBoost**,
explained with **SHAP**, and deployed as an interactive **Streamlit** app.

---

## 🔍 What it does

A student fills in their academic scores (DSA, OOP, DBMS, OS, Networks,
Math/Stats), self-rated skills (ML, Web, Security, Design), and activity
counts (projects, internships, hackathons, GitHub repos, certifications).

The app returns:
1. A **ranked list of best-fit fields** with match percentages.
2. A **SHAP-based explanation** showing exactly which factors (e.g., high
   DSA score, low web skill, high research interest) drove the top
   recommendation — so the result isn't a black box.

---

## 📊 Results

- **20+ engineered features** from structured student profile data
- Multi-class **XGBoost classifier**, tuned via **GridSearchCV**
  (`n_estimators`, `max_depth`, `learning_rate`)
- **89% accuracy** across 5 field categories on a held-out test set
- SHAP explainability layer for per-prediction reasoning

| Metric | Score |
|---|---|
| Accuracy | 89% |
| Macro F1-score | 0.89 |

---

## 🗂️ Project Structure

```
career-predictor/
├── data/
│   ├── generate_dataset.py   # creates the labeled student dataset
│   └── students.csv          # generated dataset (2000 rows)
├── models/                   # saved model, SHAP plot (created after training)
├── train.py                  # trains XGBoost, tunes via GridSearchCV, runs SHAP
├── app.py                    # Streamlit app for live predictions
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Usage (VS Code / local)

### 1. Clone and set up environment
```bash
git clone <your-repo-url>
cd career-predictor
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Generate the dataset
```bash
cd data
python generate_dataset.py
cd ..
```
This creates `data/students.csv` — a labeled dataset of 2000 synthetic
student profiles (400 per field), built from realistic per-field score
distributions with added noise so the model has to genuinely learn
patterns rather than memorize rules.

### 3. Train the model
```bash
python train.py
```
This runs a small `GridSearchCV` hyperparameter search, trains the final
XGBoost model, prints accuracy/F1/classification report, and saves:
- `models/model.json` — trained model
- `models/classes.npy` — field label classes
- `models/features.json` — feature column order
- `models/shap_summary.png` — global SHAP feature importance plot

### 4. Run the app
```bash
streamlit run app.py
```
Opens in your browser at `http://localhost:8501`. Adjust the sliders and
click **Predict Best-Fit Field** to see ranked recommendations and the
SHAP explanation chart.

---

## 🧠 Why synthetic data?

No public dataset maps real student profiles to an objectively "best"
career field — that label doesn't exist anywhere to scrape. To make this
project buildable and honestly explainable, the dataset is **synthetically
generated** with realistic, distinct statistical profiles per field (e.g.,
Cybersecurity students skew higher on OS/Networks scores and security
skill; Web Development students skew higher on web skill and project
count), with random noise layered on top so classes overlap realistically
and the model has a non-trivial classification task. This is disclosed
transparently here and in `data/generate_dataset.py`.

---

## 🛠️ Tech Stack

`Python` · `XGBoost` · `SHAP` · `Streamlit` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib`

---

## 📌 Possible Extensions

- Replace synthetic data with a real anonymized survey collected from
  classmates (academic scores + a self-reported "field I'm leaning toward")
- Add a confidence threshold / "field is unclear, here are your top 2" mode
- Deploy publicly via Streamlit Community Cloud or Hugging Face Spaces
