# ACTG 175: Predicting an AIDS-Defining Event or Death with Five Classifiers

**M.Tech (AIML) · Machine Learning · Assignment 2**
Mani Rathinam Subbaiah · BITS ID 2026AJ05440

---

## a. Problem Statement

The AIDS Clinical Trials Group Study 175 (ACTG 175) randomised HIV-infected adults with CD4 counts of 200–500 cells/mm³ to four antiretroviral regimens.
For each participant we know baseline clinical, demographic and treatment-history variables, plus CD4/CD8 counts measured at about 20 weeks.

**Task:** a **binary classification** of whether a participant **had an AIDS-defining event or died during follow-up** (`cid = 1`) or was censored without an event (`cid = 0`).
A reliable classifier helps flag the patients at highest risk, who need closer monitoring or a change of therapy.

Five classifiers are trained on the same data: Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and a Support Vector Machine.
Each is evaluated on a held-out test set with Accuracy, AUC, Precision, Recall, F1 and MCC.
Only about 1 in 4 patients has an event, so **MCC** (Matthews Correlation Coefficient) is the headline metric: it stays near 0 for a model that ignores the minority class, even when its accuracy looks high.
The models are served through an interactive Streamlit app that scores an uploaded test file **without retraining**.

> Note: `cd420`, `cd820` and `offtrt` are recorded after baseline. The task is therefore a retrospective classification of the trial outcome, not a baseline-only prognostic tool.

## b. Dataset Description

| Item | Detail |
|---|---|
| Name | AIDS Clinical Trials Group Study 175 |
| Repository | UCI Machine Learning Repository, dataset id 890: <https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175> |
| Original study | Hammer S.M. et al. (1996), *A trial comparing nucleoside monotherapy with combination therapy in HIV-infected adults with CD4 cell counts from 200 to 500 per cubic millimeter*, New England Journal of Medicine 335(15): 1081–1090 |
| Licence | UCI refers to the original source, the CRAN package `speff2trial` (GPL-2) |
| Task type | Binary classification |
| Instances | **2,139** patients (raw and after cleaning: no missing values, no duplicate rows) |
| Columns in raw file | 25 = 1 patient ID + 23 candidate features + 1 target |
| Features used | **21** (8 continuous, 11 binary, 2 nominal) |
| Target | `cid`: 1 = AIDS-defining event or death, 0 = censored |

**Class distribution**

| `cid` | Meaning | Patients | Share |
|---|---|---|---|
| 0 | No event (censored) | 1,618 | 75.6% |
| 1 | AIDS-defining event or death | 521 | 24.4% |

**Feature dictionary (21 inputs)**

| Column | Type | Meaning |
|---|---|---|
| age | continuous | Age in years at baseline |
| wtkg | continuous | Weight in kg at baseline |
| karnof | continuous | Karnofsky performance score (0–100; higher = better daily functioning) |
| preanti | continuous | Days of antiretroviral therapy before the trial |
| cd40 | continuous | CD4 count at baseline (cells/mm³) |
| cd420 | continuous | CD4 count at 20 ± 5 weeks |
| cd80 | continuous | CD8 count at baseline (cells/mm³) |
| cd820 | continuous | CD8 count at 20 ± 5 weeks |
| hemo | binary | Haemophilia (0 = no, 1 = yes) |
| homo | binary | Homosexual activity (0 = no, 1 = yes) |
| drugs | binary | History of intravenous drug use (0 = no, 1 = yes) |
| oprior | binary | Non-ZDV antiretroviral therapy before the trial |
| z30 | binary | ZDV in the 30 days before the trial |
| race | binary | 0 = white, 1 = non-white |
| gender | binary | 0 = female, 1 = male |
| str2 | binary | Antiretroviral history (0 = naive, 1 = experienced) |
| symptom | binary | Symptomatic at baseline (0 = no, 1 = yes) |
| treat | binary | 0 = ZDV only, 1 = any other arm |
| offtrt | binary | Went off treatment before 96 ± 5 weeks |
| trt | nominal | Treatment arm (0 = ZDV only, 1 = ZDV + ddI, 2 = ZDV + zalcitabine, 3 = ddI only) |
| strat | nominal | Prior-therapy stratum (1 = naive, 2 = >1 to 52 weeks, 3 = >52 weeks) |

**Columns removed before modelling**

| Column | Reason |
|---|---|
| `pidnum` | Patient identifier, carries no clinical signal |
| `time` | Days until the event or censoring. It is only known after the outcome; on its own it separates the classes with ROC-AUC 0.888, so keeping it would leak the answer |
| `zprior` | Constant: every patient has the value 1 |

**Preprocessing**, inside each model's pipeline and fitted on training data only:

- Continuous columns: median imputation, then standard scaling.
- Binary columns: most-frequent imputation, kept as 0/1.
- `trt` and `strat`: one-hot encoding.

**Train / test split:** stratified 80 / 20 with random seed 5440.

- Train: 1,711 rows (24.37% events).
- Test: 428 rows (104 events, 24.30%).

The test rows are saved as [`test_data.csv`](test_data.csv) (21 feature columns + `cid`). This is the file used for every metric below, and the file to upload to the app.

## c. GitHub Repository Link

**<https://github.com/Investrader786/actg175-aids-event-prediction>**

## d. Models Used

All five models share the same preprocessing. Hyper-parameters were tuned with `GridSearchCV`, 5-fold stratified cross-validation on the **training set only**, choosing the setting with the best MCC.

| Model | Implementation | Final hyper-parameters | CV MCC (mean ± std) |
|---|---|---|---|
| Logistic Regression | `LogisticRegression` | C = 3, class_weight = balanced, max_iter = 5000 | 0.3554 ± 0.0355 |
| Decision Tree | `DecisionTreeClassifier` | max_depth = 2, min_samples_leaf = 1, criterion = gini, class_weight = balanced | 0.3660 ± 0.0433 |
| KNN | `KNeighborsClassifier` | n_neighbors = 11, weights = distance, p = 1 (Manhattan) | 0.2312 ± 0.0278 |
| Naive Bayes | `GaussianNB`: the features are continuous or binary, not counts, so the Gaussian variant was chosen | var_smoothing = 1e-11 | 0.3186 ± 0.0447 |
| Support Vector Machine | `SVC`, RBF kernel, `probability=True` for AUC | C = 1, gamma = 0.03, class_weight = balanced | 0.3604 ± 0.0381 |

### Comparison Table

Evaluation on the held-out test set (428 patients):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6706 | 0.7532 | 0.4011 | 0.7212 | 0.5155 | 0.3247 |
| Decision Tree | 0.7266 | 0.6983 | 0.4435 | 0.4904 | 0.4658 | 0.2833 |
| KNN | 0.7593 | 0.6029 | 0.5217 | 0.1154 | 0.1890 | 0.1549 |
| Naive Bayes | 0.6846 | 0.6693 | 0.3719 | 0.4327 | 0.4000 | 0.1887 |
| Support Vector Machine | 0.6729 | 0.7301 | 0.3953 | 0.6538 | 0.4928 | 0.2912 |

*Metric convention:* binary task, positive class = 1 (AIDS event / death).

- Precision, Recall and F1 use `average='binary'` with `pos_label=1`.
- AUC is the ROC-AUC of the predicted probability of class 1.
- MCC is `matthews_corrcoef`.

All values are rounded to 4 decimals. Uploading `test_data.csv` to the app reproduces them exactly.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression gained the ability to identify more patients who actually experienced an AIDS-defining event or death: its Recall is 0.7212, correctly identifying 75 out of 104 actual event patients. It gave up overall Accuracy, which dropped to 0.6706 compared with the baseline of 0.7570, because it generated 112 false alarms. By using balanced class weights, the model gives greater importance to the minority event class. This is useful when missing a high-risk patient is more costly than flagging a patient unnecessarily. |
| Decision Tree | The biggest advantage is its simplicity and interpretability: it uses only one feature, predicting an event if CD4 count at 20 weeks (`cd420`) is at or below 273.5. However, it may miss important information from other clinical variables. It achieves an AUC of just 0.6983 and identifies only 51 of the 104 actual events (Recall 0.4904). |
| KNN | KNN achieves the highest Accuracy of 0.7593, but this is misleading because about 75.7% of patients belong to the no-event class. It predicts only 23 patients as having an event, correctly identifies just 12 of them, and misses 92 of the 104 actual event patients (Recall 0.1154, MCC 0.1549). Accuracy alone is not a reliable measure for imbalanced datasets. |
| Naive Bayes | Naive Bayes assumes input features are conditionally independent given the target class, but several features here are strongly correlated: `str2` and `strat` (Spearman 0.92), and `cd80` and `cd820` (0.74). It may therefore count the same evidence multiple times, which can distort the predicted probabilities and reduce performance. It achieves an MCC of only 0.1887, indicating relatively weak predictive performance. |
| Support Vector Machine | The RBF kernel allows curved decision boundaries, but this extra flexibility did not beat Logistic Regression on the test set: MCC 0.2912 vs 0.3247, AUC 0.7301 vs 0.7532, Recall 0.6538 vs 0.7212, F1 0.4928 vs 0.5155. SVM also has a larger train-to-test MCC drop (0.4295 to 0.2912, against 0.3716 to 0.3247), which suggests it overfits more. |
| Overall, Winner for your dataset? | **Logistic Regression.** The dataset is imbalanced, with only 24.3% of test patients experiencing an event, so MCC is more informative than Accuracy because it considers all four confusion matrix outcomes. It has the highest test MCC (0.3247), Recall (0.7212), AUC (0.7532) and F1 (0.5155), and the smallest train-to-test MCC drop. The trade-off is the lowest Accuracy (0.6706) and the most false alarms (112). Cross-validation is close (Decision Tree 0.3660, SVM 0.3604, Logistic Regression 0.3554, within one standard deviation), so the win is not decisive, and with a test MCC of 0.3247 it is a useful risk-flagging model rather than a definitive clinical decision-making system. |

---
## Live Streamlit App

**<https://actg175-outcome-evaluator-mani.streamlit.app>**

The app has three tabs:

1. **Evaluate selected model:** the six metrics, the confusion matrix, the classification report, and downloadable predictions.
2. **Compare all five models:** a metric table, a bar chart and ROC curves.
3. **About the data & models.**

Upload `test_data.csv`, or switch on *Use the bundled test_data.csv*. The app only loads the saved pipelines from `model/` and never retrains.

## Repository Structure

```
actg175-aids-event-prediction/
├── app.py                         # Streamlit app (loads saved models, never retrains)
├── actg_eval.py                   # shared schema + metric definitions (used by notebook and app)
├── actg175_model_training.ipynb   # full workflow: EDA → cleaning → split → tuning → evaluation → saving
├── requirements.txt               # pinned dependencies (installable on Python 3.10–3.13)
├── test_data.csv                  # held-out test set: 428 rows, 21 features + cid
├── data/actg175_raw.csv           # raw UCI table (2,139 rows)
├── model/
│   ├── logistic_regression.joblib # each file = fitted preprocessing + classifier pipeline
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   └── model_card.json            # schema, seed, split, versions, tuned settings, test metrics
├── results/                       # metrics_comparison.csv/.md, train_metrics.csv, cv_summary.csv, confusion_matrices.json
├── scripts/verify_artifacts.py    # reloads every model and checks it reproduces the reported metrics
└── .streamlit/config.toml         # app theme and upload limit
```

## How to Run Locally (or on the BITS Virtual Lab)

```bash
pip install -r requirements.txt          # Python 3.10 – 3.13
python scripts/verify_artifacts.py       # optional: confirm the saved models reproduce the table above
streamlit run app.py                     # opens the app at http://localhost:8501
```

To retrain from scratch, open `actg175_model_training.ipynb` from the repository root and choose **Restart & Run All** (Jupyter is needed in addition to the packages above).
The run is deterministic (seed 5440), so it regenerates the same models and the same metrics.

## Reproducibility and Metric Definitions

- **Seed:** 5440 for the split, the CV folds and the models.
- **Library versions:** pinned in `requirements.txt`; the exact training versions are recorded in `model/model_card.json`.
- **One source of truth:** `actg_eval.score_predictions` defines all six metrics. The notebook and the app both call it, so the README table and the app always agree.
- **Accuracy:** share of correct predictions.
- **AUC:** area under the ROC curve of P(cid = 1).
- **Precision:** TP / (TP + FP) for the event class.
- **Recall:** TP / (TP + FN) for the event class.
- **F1:** harmonic mean of Precision and Recall.
- **MCC:** correlation between predicted and actual labels, from −1 to +1. It is 0 for chance-level or single-class predictions.
