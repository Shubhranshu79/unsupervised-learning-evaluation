# Week 3 - Unsupervised Learning & Model Evaluation

Clustering and model evaluation on the cleaned loan applicant dataset from Week 1
(Python for Machine Learning and Data Preprocessing).

- **Part A - Unsupervised learning:** PCA (dimensionality reduction), K-Means and Hierarchical clustering,
  cluster interpretation.
- **Part B - Model evaluation:** cross-validation, confusion matrix, precision, recall, F1-score, ROC-AUC,
  hyperparameter tuning (GridSearchCV and RandomizedSearchCV), validation curve and decision threshold.
- **Part C - Why the models perform differently:** feature importance across models (coefficients, impurity,
  permutation), feature ablation and an engineered feature, learning curves, a corrected paired significance
  test, a noise ceiling and an error analysis.

## Results

### Part A - Clustering

The approval label is **not** used for clustering. Features: age, education, income, credit score, loan amount,
loan term (scaled).

| Finding | Result |
|---|---|
| PCA | PC1 + PC2 keep 59% of the variance; 5 of 6 components needed for 95% |
| Best k (silhouette) | k = 2 (silhouette 0.234, weak structure) |
| K-Means vs Hierarchical (Ward) | Adjusted Rand Index 0.336 (moderate agreement) |

| Cluster | Size | Avg age | Avg income (Rs.) | Credit score | Avg loan (Rs.) | Approval rate |
|---|---|---|---|---|---|---|
| 0 - older, higher income | 253 | 51.7 | 954,572 | 625 | 911,530 | 68.4% |
| 1 - younger, lower income | 262 | 34.4 | 608,361 | 595 | 474,332 | 54.6% |

The two clusters split one continuous group in two halves (low silhouette), so they are best used as descriptive
segments, not as sharply separate populations.

### Part B - Model evaluation

5-fold cross-validation repeated 3 times (15 folds), then a final check on an untouched 103-row test set.

| Model | CV ROC-AUC (mean +/- std) | Test accuracy | Test precision | Test recall | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline (always approve) | 0.500 +/- 0.000 | 0.612 | 0.612 | 1.000 | 0.759 | 0.500 |
| Logistic Regression (tuned) | 0.732 +/- 0.065 | 0.718 | 0.736 | 0.841 | 0.785 | 0.756 |
| KNN (tuned) | 0.654 +/- 0.063 | 0.650 | 0.671 | 0.841 | 0.746 | 0.749 |
| Random Forest (tuned) | 0.711 +/- 0.055 | 0.709 | 0.714 | 0.873 | 0.786 | 0.755 |

(CV scores are for the default settings; tuned CV AUC: Logistic Regression 0.738, Random Forest 0.724, KNN 0.717.)

**Findings**
- Logistic Regression is the best model by cross-validated AUC. Confusion matrix on the test set:
  TP = 53, TN = 21, FP = 19, FN = 10.
- The always-approve baseline already gets accuracy 0.61 and F1 0.76, so **ROC-AUC** is the metric that shows
  real skill (0.76 vs 0.50).
- The untuned Random Forest overfits: training AUC 1.00 vs validation AUC 0.71.
- The decision threshold is a business choice: 0.4 gives recall 0.94, 0.6 gives precision 0.78.
- With only 515 rows (synthetic data), differences between the tuned models are within noise.

### Part C - Why do the models perform differently?

| Question | Finding |
|---|---|
| Which features do the models use? | All three rely on **credit score** (test AUC drops by 0.18-0.22 when it is shuffled). Beyond that they disagree: Logistic Regression uses income and loan size together (a loan-to-income signal), KNN and Random Forest use Age as a stand-in for income. Differences below the top feature are mostly within noise. |
| Do the importance measures agree? | Only partly. Rank agreement between models is low (Spearman 0.16-0.53). Random Forest impurity importance overstates `Loan_Term_Months` and `Education`; a large coefficient (e.g. the Unemployed dummy) is not the same as an important feature. |
| Why is Logistic Regression best? | The data are roughly linear in a few features; regularisation ignores useless columns (AUC 0.731-0.746 on every feature set); training and validation curves meet, so it needs little data. |
| Why is KNN last? | Distance counts all 14 features equally (AUC 0.698 with all features, 0.713 with 3 financial ones); it memorises (training AUC 1.00); it is weakest in sparse regions. |
| Why does Random Forest sit in the middle? | Flexible but data-hungry: its validation AUC is still rising at 329 training rows (0.676 to 0.728). |
| Is the ranking real? | **Not statistically established.** Logistic Regression leads by +0.016 (vs Random Forest) and +0.037 (vs KNN) AUC, corrected paired p = 0.23 and 0.24. |
| Could a better model do much better? | Probably not. The true generating rule gives AUC 0.744 on all rows; Logistic Regression reaches 0.739 out-of-fold. 25.6% of applicants are wrong for all three models, and within credit-score bands the models barely beat a constant guess. |



