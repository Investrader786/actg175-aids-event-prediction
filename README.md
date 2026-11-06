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
| Logistic Regression | Best overall: it has the highest AUC (0.7532), Recall (0.7212), F1 (0.5155) and MCC (0.3247), and catches 75 of the 104 actual events. Its Accuracy (0.6706) is the lowest of the five and below the 0.7570 scored by always predicting "no event", because balanced class weights accept more false alarms (112) in exchange for fewer missed events; without the weights, Recall falls to 0.2596 and MCC to 0.2582. It generalises best, with the smallest train-to-test MCC drop (0.3716 to 0.3247). The largest coefficient by far is `cd420` (-1.09), so a higher CD4 count at 20 weeks lowers the predicted risk. |
| Decision Tree | Cross-validation chose a very shallow tree: CV MCC was 0.3660 at depth 2 and 0.33-0.34 at every greater depth. The saved tree uses only `cd420`, and its predictions reduce to one rule: predict an event when `cd420` is at or below 273.5 cells/mm³. This gives the second-best Accuracy (0.7266) and Precision (0.4435) but catches only 51 of the 104 events (Recall 0.4904). With just 4 leaves it outputs 4 distinct probabilities, so it ranks patients coarsely and its AUC is only 0.6983. |
| KNN | It has the highest Accuracy (0.7593) and Precision (0.5217), but both are misleading: it predicts an event for only 23 of the 428 patients and catches just 12 of the 104 actual events (Recall 0.1154), so its Accuracy barely beats the 0.7570 baseline. It has the lowest AUC (0.6029), F1 (0.1890) and MCC (0.1549). KNN has no class-weight option, so the majority class tends to dominate the neighbourhoods, and no value of k tested raised CV MCC above 0.2312. Its training scores of 1.0000 come from distance weighting, where each training point is its own nearest neighbour, and do not show real skill. |
| Naive Bayes | Weak overall: it ranks fourth on AUC (0.6693), Recall (0.4327), F1 (0.4000) and MCC (0.1887), and last on Precision (0.3719). Its independence assumption does not hold here: related features are strongly correlated, for example `str2` and `strat` (Spearman 0.92) and `cd80` and `cd820` (0.74), so the same evidence is counted more than once, which likely hurts it. GaussianNB also treats the 0/1 features as normally distributed. Tuning did not help, as CV MCC stayed at 0.3186 for every `var_smoothing` value except the largest. |
| Support Vector Machine | Second best: it ranks second on AUC (0.7301), Recall (0.6538), F1 (0.4928) and MCC (0.2912), catching 68 of the 104 events with 104 false alarms. Like Logistic Regression it uses balanced class weights, so its Accuracy (0.6729) is also below the 0.7570 baseline. The RBF kernel shows no clear gain over the linear model, which suggests the signal is mostly linear, and it overfits more (train-to-test MCC drop 0.4295 to 0.2912). It is also the slowest model to fit, because `probability=True` adds internal cross-validation for calibration. |
| Overall, Winner for your dataset? | **Logistic Regression.** MCC is the primary metric because only 24.3% of test patients have an event, so Accuracy rewards models that rarely predict one: KNN has the best Accuracy but the worst MCC. Logistic Regression is best on MCC, AUC, Recall and F1, has the smallest train-to-test drop and is easy to interpret through its coefficients; the SVM is the runner-up. The trade-off is the lowest Accuracy and the most false alarms (112), which is acceptable when flagging at-risk patients matters more than raw Accuracy. The win is not decisive: in cross-validation the top three were within one standard deviation of each other (Decision Tree 0.3660, SVM 0.3604, Logistic Regression 0.3554), and skill is only moderate, with every test MCC below 0.33. |

---

## Live Streamlit App

**<https://YOUR-APP-NAME.streamlit.app>**

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
