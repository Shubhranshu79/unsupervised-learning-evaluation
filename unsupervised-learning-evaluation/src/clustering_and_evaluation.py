"""
Week 3 Task - Unsupervised Learning & Model Evaluation
======================================================
Part A  Unsupervised learning : PCA (dimensionality reduction), K-Means and Hierarchical clustering
Part B  Model evaluation      : cross-validation, confusion matrix, precision, recall, F1-score,
                                ROC-AUC, hyperparameter tuning, decision threshold
Part C  Why models differ     : feature importance across models, feature ablation, feature engineering,
                                learning curves, paired significance tests, noise ceiling, error analysis
Data    : data/loan_applicants_clean.csv (output of the Week 1 preprocessing task)

Run from the repository root:   python3 src/clustering_and_evaluation.py
Creates: results/*.csv, results/charts/*.png, results/log.json, models/best_classifier.joblib
"""
import io
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import linear_sum_assignment
from scipy.stats import randint
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from scipy.stats import t as t_dist
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, classification_report, confusion_matrix,
                             davies_bouldin_score, f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve, silhouette_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict, cross_val_score, cross_validate,
                                     learning_curve, train_test_split, validation_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

OUT = "results"
os.makedirs(f"{OUT}/charts", exist_ok=True)
os.makedirs("models", exist_ok=True)
pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 30)


# ===== SETUP: capture printed output per step (used for the PDF report) =====
class Tee:
    def __init__(self):
        self.buf = io.StringIO()

    def write(self, s):
        self.buf.write(s)
        sys.__stdout__.write(s)

    def flush(self):
        sys.__stdout__.flush()


tee = Tee()
sys.stdout = tee
LOG, _current = {}, None


def step(n):
    global _current
    if _current is not None:
        LOG[str(_current)] = tee.buf.getvalue().rstrip()
    tee.buf = io.StringIO()
    _current = n


# ===== SETUP: chart style and plotting helpers =====
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, VIOLET = "#e87ba4", "#6250d6"
CLUSTER_COLORS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e2e1dc", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "savefig.dpi": 170,
})


def save(fig, name):
    fig.savefig(f"{OUT}/charts/{name}.png", bbox_inches="tight")
    plt.close(fig)


def plot_pca_variance(evr):
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    x = np.arange(1, len(evr) + 1)
    ax.bar(x, evr, color=BLUE, width=0.55, label="Variance of each component")
    ax.plot(x, evr.cumsum(), color=ORANGE, marker="o", linewidth=2, label="Cumulative variance")
    for xi, v in zip(x, evr):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i}" for i in x])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Share of total variance")
    ax.set_title("PCA: how much information each component keeps")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    save(fig, "01_pca_variance")


def plot_k_selection(k_table, best_k):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    ks = k_table.index.to_numpy()
    axes[0].plot(ks, k_table["Inertia"], color=BLUE, marker="o", linewidth=2)
    axes[0].set_title("Elbow method")
    axes[0].set_ylabel("Inertia (lower = tighter clusters)")
    axes[1].plot(ks, k_table["Silhouette"], color=BLUE, marker="o", linewidth=2)
    axes[1].plot([best_k], [k_table.loc[best_k, "Silhouette"]], color=ORANGE, marker="o", markersize=10,
                 linestyle="none", label=f"Best: k = {best_k}")
    axes[1].set_title("Silhouette score")
    axes[1].set_ylabel("Silhouette (higher = better)")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel("Number of clusters (k)")
        ax.set_xticks(ks)
    fig.tight_layout()
    save(fig, "02_kmeans_selection")


def plot_dendrogram(Z, k):
    t = (Z[-(k - 1), 2] + Z[-k, 2]) / 2 if k > 1 else Z[-1, 2] + 1
    hierarchy.set_link_color_palette(CLUSTER_COLORS[:k])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    dendrogram(Z, truncate_mode="lastp", p=30, no_labels=True, color_threshold=t,
               above_threshold_color=INK2, ax=ax)
    ax.axhline(t, color=INK2, linestyle="--", linewidth=1)
    ax.text(0.5, t, f"  cut here: {k} clusters", va="bottom", fontsize=8, color=INK2)
    ax.set_ylabel("Distance at which clusters merge")
    ax.set_xlabel("Applicants (grouped into the last 30 merged branches)")
    ax.set_title("Hierarchical clustering: dendrogram (Ward linkage)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "03_dendrogram")
    hierarchy.set_link_color_palette(None)


def plot_clusters(pc, km_labels, hc_labels, evr, k):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharex=True, sharey=True)
    for ax, labels, title in zip(axes, (km_labels, hc_labels), ("K-Means", "Hierarchical (Ward)")):
        for c in range(k):
            m = labels == c
            ax.scatter(pc[m, 0], pc[m, 1], s=16, color=CLUSTER_COLORS[c], alpha=0.75,
                       edgecolor=SURFACE, linewidth=0.4, label=f"Cluster {c}")
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({evr[0] * 100:.0f}% of variance)")
    axes[0].set_ylabel(f"PC2 ({evr[1] * 100:.0f}% of variance)")
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    save(fig, "04_clusters_pca")


def plot_profiles(profile_z):
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#eeedea", ORANGE])
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    im = ax.imshow(profile_z.T.to_numpy(), cmap=cmap, vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(profile_z.shape[0]))
    ax.set_xticklabels([f"Cluster {i}" for i in profile_z.index])
    ax.set_yticks(range(profile_z.shape[1]))
    ax.set_yticklabels(profile_z.columns)
    for i in range(profile_z.shape[1]):
        for j in range(profile_z.shape[0]):
            ax.text(j, i, f"{profile_z.iat[j, i]:+.2f}", ha="center", va="center", fontsize=8.5, color=INK)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.outline.set_visible(False)
    ax.set_title("Cluster profiles (average z-score; orange = above average)")
    fig.tight_layout()
    save(fig, "05_cluster_profiles")


def plot_cv(cv_summary):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    names = list(cv_summary.index)[::-1]
    for ax, (m, title) in zip(axes, (("roc_auc", "ROC-AUC"), ("f1", "F1 score"))):
        mean = cv_summary.loc[names, f"{m}_mean"].to_numpy()
        std = cv_summary.loc[names, f"{m}_std"].to_numpy()
        ax.errorbar(mean, range(len(names)), xerr=std, fmt="o", color=BLUE, ecolor=BLUE, capsize=3,
                    markersize=6, linewidth=1.5)
        for y, (a, s) in enumerate(zip(mean, std)):
            ax.text(a, y + 0.16, f"{a:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.set_xlim(0.45, 0.95)
        ax.set_title(f"{title} (mean and spread, 15 folds)", fontsize=9)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "06_cross_validation")


def plot_validation(depths, train_s, val_s):
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    for s, color, lab in ((train_s, BLUE, "Training AUC"), (val_s, ORANGE, "Cross-validation AUC")):
        mu, sd = s.mean(axis=1), s.std(axis=1)
        ax.plot(depths, mu, color=color, marker="o", linewidth=2, label=lab)
        ax.fill_between(depths, mu - sd, mu + sd, color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel("Random Forest max_depth")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Validation curve: where the model starts to overfit")
    ax.set_xticks(depths)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    save(fig, "07_validation_curve")


def plot_confusion(preds, y_true):
    cmap = LinearSegmentedColormap.from_list("seq", [SURFACE, BLUE])
    fig, axes = plt.subplots(1, len(preds), figsize=(6.6, 2.5))
    for ax, (name, p) in zip(axes, preds.items()):
        cm = confusion_matrix(y_true, p)
        ax.imshow(cm, cmap=cmap)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                        color="white" if cm[i, j] > 0.7 * cm.max() else INK)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Rejected", "Approved"], fontsize=7)
        ax.set_yticklabels(["Rejected", "Approved"], fontsize=7)
        ax.set_xlabel("Predicted", fontsize=7.5)
        ax.set_title(name, fontsize=8.5)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("Actual", fontsize=7.5)
    fig.tight_layout()
    save(fig, "08_confusion_matrices")


def plot_roc(probas, y_true, aucs):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for (name, p), c in zip(probas.items(), (BLUE, ORANGE, AQUA)):
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, color=c, linewidth=2, label=f"{name} (AUC {aucs[name]:.2f})")
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle="--", label="Random guess (AUC 0.50)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves of the tuned models (test set)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "09_roc_curves")


def plot_threshold(thr, prec, rec, f1):
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    ax.plot(thr, prec, color=BLUE, linewidth=2, label="Precision")
    ax.plot(thr, rec, color=ORANGE, linewidth=2, label="Recall")
    ax.plot(thr, f1, color=AQUA, linewidth=2, label="F1 score")
    ax.axvline(0.5, color=INK2, linestyle="--", linewidth=1)
    ax.text(0.505, 0.05, "default threshold 0.5", fontsize=8, color=INK2)
    ax.set_xlabel("Decision threshold (probability needed to predict 'approved')")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall trade-off when the threshold moves")
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 0.12))
    fig.tight_layout()
    save(fig, "10_threshold_tradeoff")


MODEL_COLORS = {"Logistic Regression": BLUE, "KNN": ORANGE, "Random Forest": AQUA}


def plot_perm_importance(perm, top=8):
    order = pd.concat([d["mean"] for d in perm.values()], axis=1).mean(axis=1).sort_values(ascending=False).index[:top]
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    h, y = 0.26, np.arange(len(order))[::-1]
    for i, (name, d) in enumerate(perm.items()):
        ax.barh(y + (1 - i) * h, d.loc[order, "mean"], h, xerr=d.loc[order, "std"], color=MODEL_COLORS[name],
                label=name, error_kw=dict(ecolor=INK2, lw=0.8, capsize=1.5))
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("Drop in test ROC-AUC when the feature is shuffled (bigger = model relies on it more)")
    ax.set_title("Permutation importance: which features each model relies on")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "11_feature_importance_models")


def plot_ablation(tab):
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    x = np.arange(len(tab))
    for i, name in enumerate(tab.columns):
        ax.plot(x, tab[name], color=MODEL_COLORS[name], marker="o", linewidth=2, label=name)
        for xi, v in zip(x, tab[name]):
            up = i == 0                                   # Logistic Regression labels above, the others below
            ax.text(xi, v + (0.005 if up else -0.005), f"{v:.3f}", ha="center", va="bottom" if up else "top",
                    fontsize=6.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" + ", "\n+ ") for s in tab.index], fontsize=7)
    ax.set_ylabel("Cross-validated ROC-AUC")
    ax.set_title("Same models, different feature sets")
    ax.set_ylim(0.645, 0.762)
    ax.grid(axis="x", visible=False)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "12_feature_ablation")


def plot_learning(lc):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
    for ax, (name, (n_tr, tr, va)) in zip(axes, lc.items()):
        for s, col, lab, ls in ((tr, BLUE, "Training", "--"), (va, ORANGE, "Validation", "-")):
            mu, sd = s.mean(axis=1), s.std(axis=1)
            ax.plot(n_tr, mu, color=col, marker="o", linewidth=2, linestyle=ls, label=lab)
            ax.fill_between(n_tr, mu - sd, mu + sd, color=col, alpha=0.12, linewidth=0)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Training rows")
    axes[0].set_ylabel("ROC-AUC")
    axes[1].legend(frameon=False, fontsize=8, loc="center", ncol=2)    # the empty middle of the KNN panel
    fig.tight_layout()
    save(fig, "13_learning_curves")


def plot_errors(err_credit, err_lti, base_credit, base_lti):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    for ax, tab, base, title in ((axes[0], err_credit, base_credit, "Error rate by credit-score quartile"),
                                 (axes[1], err_lti, base_lti, "Error rate by loan-to-income quartile")):
        x, w = np.arange(len(tab)), 0.26
        for i, name in enumerate(tab.columns):
            ax.bar(x + (i - 1) * w, tab[name] * 100, w, color=MODEL_COLORS[name], label=name)
        ax.hlines(base.to_numpy() * 100, x - 0.42, x + 0.42, color=INK, linewidth=1.6, linestyle=":",
                  label="Best constant guess in the band")
        ax.set_xticks(x)
        ax.set_xticklabels(tab.index, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Wrongly classified (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[:4], l[:4], loc="lower center", ncol=4, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "14_error_analysis")


# ===== STEP 1: Load the data and choose the clustering features =====
step(1)
df = pd.read_csv("data/loan_applicants_clean.csv")
print("Shape:", df.shape)
# Clustering is unsupervised: the label (Approved) is NOT used. Only the six numeric / ordinal
# columns that describe the applicant's profile are used; 0/1 dummy columns would dominate distances.
features = ["Age", "Education", "Annual_Income", "Credit_Score", "Loan_Amount", "Loan_Term_Months"]
X_scaled = StandardScaler().fit_transform(df[features])   # clustering needs all features on one scale
print("Clustering features:", features)
print("Scaled matrix shape:", X_scaled.shape)
print("\nAverage and spread of the scaled features (should be 0 and 1):")
print(pd.DataFrame(X_scaled, columns=features).agg(["mean", "std"]).round(2))

# ===== STEP 2: Dimensionality reduction with PCA =====
step(2)
pca = PCA(n_components=len(features), random_state=42).fit(X_scaled)
evr = pca.explained_variance_ratio_
print(pd.DataFrame({"Explained": evr.round(3), "Cumulative": evr.cumsum().round(3)},
                   index=[f"PC{i + 1}" for i in range(len(evr))]))
n95 = int(np.argmax(evr.cumsum() >= 0.95)) + 1
print(f"\nComponents needed to keep 95% of the information: {n95} of {len(features)}")
print("\nWhat PC1 and PC2 are made of (loadings):")
print(pd.DataFrame(pca.components_[:2].T, index=features, columns=["PC1", "PC2"]).round(2))
pc2 = PCA(n_components=2, random_state=42).fit_transform(X_scaled)   # 2-D view used for plotting
plot_pca_variance(evr)

# ===== STEP 3: K-Means clustering and choosing k =====
step(3)
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    rows.append({"k": k, "Inertia": km.inertia_,
                 "Silhouette": silhouette_score(X_scaled, km.labels_),
                 "Davies_Bouldin": davies_bouldin_score(X_scaled, km.labels_)})
k_table = pd.DataFrame(rows).set_index("k")
print(k_table.round(3))
best_k = int(k_table["Silhouette"].idxmax())          # highest silhouette = best separated clusters
km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X_scaled)
km_labels = km.labels_
print(f"\nChosen k = {best_k} (highest silhouette score)")
print("Applicants per cluster:")
print(pd.Series(km_labels).value_counts().sort_index())
plot_k_selection(k_table, best_k)
k_table.to_csv(f"{OUT}/kmeans_k_selection.csv")

# ===== STEP 4: Hierarchical clustering =====
step(4)
Z = linkage(X_scaled, method="ward")                  # bottom-up merge tree
rows = []
for method in ["ward", "complete", "average", "single"]:
    lab = AgglomerativeClustering(n_clusters=best_k, linkage=method).fit_predict(X_scaled)
    rows.append({"Linkage": method, "Silhouette": silhouette_score(X_scaled, lab),
                 "ARI_vs_KMeans": adjusted_rand_score(km_labels, lab),
                 "Smallest_cluster": int(np.bincount(lab).min())})
print(pd.DataFrame(rows).set_index("Linkage").round(3))
hc_labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X_scaled)
# Cluster numbers are arbitrary: renumber the hierarchical clusters to match K-Means
cont = pd.crosstab(km_labels, hc_labels).to_numpy()
r, c = linear_sum_assignment(-cont)
mapping = {int(ci): int(ri) for ri, ci in zip(r, c)}
hc_labels = np.array([mapping[int(l)] for l in hc_labels])
print("\nK-Means (rows) vs Hierarchical (columns):")
print(pd.crosstab(pd.Series(km_labels, name="KMeans"), pd.Series(hc_labels, name="Hierarchical")))
ari = adjusted_rand_score(km_labels, hc_labels)
print(f"\nAdjusted Rand Index between the two methods: {ari:.3f}  (1.0 = identical clusters)")
plot_dendrogram(Z, best_k)
plot_clusters(pc2, km_labels, hc_labels, evr, best_k)

# ===== STEP 5: Describe and interpret the clusters =====
step(5)
# Convert the z-scores back to real units with the means / standard deviations from Week 1 (Step 11)
SCALE = {"Age": (42.9, 12.2), "Annual_Income": (778441.6, 227053.6), "Credit_Score": (609.5, 49.0),
         "Loan_Amount": (689110.7, 365853.3), "Loan_Term_Months": (36.3, 17.0)}
profile_z = df[features].groupby(km_labels).mean()
profile = profile_z.copy()
for col, (mu, sd) in SCALE.items():
    profile[col] = profile_z[col] * sd + mu
profile.insert(0, "Size", pd.Series(km_labels).value_counts().sort_index())
profile["Approval_%"] = (df["Approved"].groupby(km_labels).mean() * 100).round(1)
print("Cluster profiles in real units (Education: 0 = High School ... 3 = PhD):")
print(profile.round({"Age": 1, "Education": 2, "Annual_Income": 0, "Credit_Score": 0,
                     "Loan_Amount": 0, "Loan_Term_Months": 1}).T)
print("\nThe approval label was NOT used for clustering, so a difference in approval rate")
print("between clusters is an independent check that the clusters are meaningful.")
plot_profiles(profile_z)
profile.round(2).to_csv(f"{OUT}/cluster_profiles.csv")
labelled = df.copy()
labelled["KMeans_Cluster"], labelled["Hierarchical_Cluster"] = km_labels, hc_labels
labelled.to_csv(f"{OUT}/clustered_data.csv", index=False)

# ===== STEP 6: Model evaluation - cross-validation =====
step(6)
X, y = df.drop(columns="Approved"), df["Approved"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train / test rows:", len(X_train), "/", len(X_test))
models = {
    "Baseline (always approve)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
}
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)   # 15 train/validate rounds
rows = []
for name, model in models.items():
    cv = cross_validate(model, X_train, y_train, cv=rcv, scoring=scoring, return_train_score=True)
    row = {"Model": name, "train_auc": cv["train_roc_auc"].mean()}
    for s in scoring:
        row[f"{s}_mean"], row[f"{s}_std"] = cv[f"test_{s}"].mean(), cv[f"test_{s}"].std()
    rows.append(row)
cv_summary = pd.DataFrame(rows).set_index("Model")
show = pd.DataFrame({s: cv_summary[f"{s}_mean"].map("{:.3f}".format) + " +/- " +
                        cv_summary[f"{s}_std"].map("{:.3f}".format) for s in scoring})
print("\nCross-validation results (mean +/- standard deviation over 15 folds):")
print(show)
print("\nOverfitting check - AUC on the training folds vs on the validation folds:")
print(pd.DataFrame({"Train_AUC": cv_summary["train_auc"], "Validation_AUC": cv_summary["roc_auc_mean"]}).round(3))
plot_cv(cv_summary)
cv_summary.round(4).to_csv(f"{OUT}/cross_validation_summary.csv")

# ===== STEP 7: Hyperparameter tuning =====
step(7)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grids = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), {"C": [0.01, 0.1, 1, 10]}),
    "KNN": (KNeighborsClassifier(), {"n_neighbors": [5, 11, 21, 31], "weights": ["uniform", "distance"]}),
    "Random Forest": (RandomForestClassifier(random_state=42),
                      {"n_estimators": [100, 200], "max_depth": [3, 4, 6, 8], "min_samples_leaf": [1, 3, 5]}),
}
tuned, rows, rf_search = {}, [], None
for name, (model, grid) in grids.items():
    gs = GridSearchCV(model, grid, cv=skf, scoring="roc_auc").fit(X_train, y_train)
    tuned[name] = gs.best_estimator_
    n_comb = len(gs.cv_results_["params"])
    rows.append({"Model": name, "Search": "Grid", "Combinations": n_comb, "Model_fits": n_comb * 5,
                 "Best_parameters": str(gs.best_params_), "CV_AUC": gs.best_score_})
    if name == "Random Forest":
        rf_search = gs
tuned_table = pd.DataFrame(rows).set_index("Model")
print("GridSearchCV tries every combination (5-fold CV, scored by ROC-AUC):")
print(tuned_table[["Combinations", "Model_fits", "CV_AUC"]].round(3))
print()
for name, row in tuned_table.iterrows():
    print(f"{name:20s} best: {row['Best_parameters']}")

res = pd.DataFrame(rf_search.cv_results_)
top = pd.concat([pd.DataFrame(list(res["params"])), res[["mean_test_score", "std_test_score"]]], axis=1)
print("\nTop 5 Random Forest combinations from the grid:")
print(top.sort_values("mean_test_score", ascending=False).head(5).round(3).to_string(index=False))

# RandomizedSearchCV: samples a fixed number of random combinations from a much wider space
rs = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    {"n_estimators": randint(50, 400), "max_depth": randint(2, 12), "min_samples_leaf": randint(1, 15),
     "max_features": ["sqrt", "log2", None]},
    n_iter=20, cv=skf, scoring="roc_auc", random_state=42).fit(X_train, y_train)
print("\nGridSearchCV vs RandomizedSearchCV for the Random Forest:")
print(pd.DataFrame({"Fits": [24 * 5, 20 * 5],
                    "Best_CV_AUC": [rf_search.best_score_, rs.best_score_]},
                   index=["GridSearchCV (24 combinations)", "RandomizedSearchCV (20 random draws)"]).round(3))
print("Randomized best parameters:", rs.best_params_)

# Validation curve: one hyperparameter at a time, training score vs validation score
depths = [1, 2, 3, 4, 6, 8, 10, 12]
tr_s, va_s = validation_curve(RandomForestClassifier(n_estimators=100, random_state=42), X_train, y_train,
                              param_name="max_depth", param_range=depths, cv=skf, scoring="roc_auc")
print("\nValidation curve (Random Forest max_depth):")
print(pd.DataFrame({"Train_AUC": tr_s.mean(axis=1), "Validation_AUC": va_s.mean(axis=1)},
                   index=pd.Index(depths, name="max_depth")).round(3))
plot_validation(depths, tr_s, va_s)
tuned_table.to_csv(f"{OUT}/tuning_results.csv")

# ===== STEP 8: Final evaluation on the test set =====
step(8)
final = {"Baseline (always approve)": models["Baseline (always approve)"].fit(X_train, y_train), **tuned}
rows, preds, probas = [], {}, {}
for name, model in final.items():
    preds[name] = model.predict(X_test)
    probas[name] = model.predict_proba(X_test)[:, 1]
    rows.append({"Model": name,
                 "Accuracy": accuracy_score(y_test, preds[name]),
                 "Precision": precision_score(y_test, preds[name], zero_division=0),
                 "Recall": recall_score(y_test, preds[name]),
                 "F1": f1_score(y_test, preds[name]),
                 "ROC_AUC": roc_auc_score(y_test, probas[name])})
test_metrics = pd.DataFrame(rows).set_index("Model")
print(test_metrics.round(3))
best = tuned_table["CV_AUC"].idxmax()                  # chosen by cross-validation, not by the test set
print(f"\nBest model (highest cross-validated AUC): {best}")
tn, fp, fn, tp = confusion_matrix(y_test, preds[best]).ravel()
print(f"\nConfusion matrix of {best}:")
print(pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual Rejected", "Actual Approved"],
                   columns=["Pred. Rejected", "Pred. Approved"]))
print(f"\nTN={tn}  FP={fp}  FN={fn}  TP={tp}")
print(f"Precision = TP / (TP + FP) = {tp}/{tp + fp} = {tp / (tp + fp):.3f}")
print(f"Recall    = TP / (TP + FN) = {tp}/{tp + fn} = {tp / (tp + fn):.3f}")
print(f"Specificity = TN / (TN + FP) = {tn}/{tn + fp} = {tn / (tn + fp):.3f}")
print("\nClassification report:")
print(classification_report(y_test, preds[best], target_names=["Rejected", "Approved"], digits=3))
tuned_only = {k: v for k, v in preds.items() if k in tuned}
plot_confusion(tuned_only, y_test)
plot_roc({k: probas[k] for k in tuned}, y_test, test_metrics["ROC_AUC"])
test_metrics.round(4).to_csv(f"{OUT}/test_metrics.csv")
joblib.dump(final[best], "models/best_classifier.joblib")

# ===== STEP 9: Decision threshold - the precision / recall trade-off =====
step(9)
p = probas[best]
rows = []
for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    pr = (p >= t).astype(int)                          # predict "approved" only if probability >= t
    rows.append({"Threshold": t, "Precision": precision_score(y_test, pr, zero_division=0),
                 "Recall": recall_score(y_test, pr), "F1": f1_score(y_test, pr),
                 "Accuracy": accuracy_score(y_test, pr)})
thr_table = pd.DataFrame(rows).set_index("Threshold")
print(f"Effect of the decision threshold for {best}:")
print(thr_table.round(3))
thr = np.linspace(0.2, 0.9, 36)
pr_c = [precision_score(y_test, (p >= t).astype(int), zero_division=0) for t in thr]
re_c = [recall_score(y_test, (p >= t).astype(int)) for t in thr]
f1_c = [f1_score(y_test, (p >= t).astype(int)) for t in thr]
plot_threshold(thr, pr_c, re_c, f1_c)
thr_table.round(4).to_csv(f"{OUT}/threshold_analysis.csv")

# ===== STEP 10: Feature importance - do the three models rely on the same features? =====
step(10)
fitted = {"Logistic Regression": tuned["Logistic Regression"], "KNN": tuned["KNN"],
          "Random Forest": tuned["Random Forest"]}
cols = list(X.columns)

# 10a. Built-in importance, read from inside the trained models (training data)
coef = pd.Series(fitted["Logistic Regression"].coef_[0], index=cols)
rf_imp = pd.Series(fitted["Random Forest"].feature_importances_, index=cols)
built_in = pd.DataFrame({"LR_coefficient": coef, "LR_rank": coef.abs().rank(ascending=False).astype(int),
                         "RF_importance": rf_imp, "RF_rank": rf_imp.rank(ascending=False).astype(int)})
print("Built-in importance (LR: coefficient per 1 std for numeric columns, per 0->1 for dummies;")
print("RF: share of the total impurity reduction):")
print(built_in.sort_values("RF_importance", ascending=False).round(3))

# 10b. Permutation importance: shuffle one column on the test set and measure how much the AUC drops.
#      Works the same way for every model, including KNN which has no built-in importance.
perm = {}
for name, model in fitted.items():
    r = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=42)
    perm[name] = pd.DataFrame({"mean": r.importances_mean, "std": r.importances_std}, index=cols)
perm_mean = pd.DataFrame({n: d["mean"] for n, d in perm.items()})
perm_mean["Average"] = perm_mean.mean(axis=1)
print("\nPermutation importance = drop in test ROC-AUC (mean of 30 shuffles):")
print(perm_mean.sort_values("Average", ascending=False).round(3))
perm_noise = pd.Series({n: d["std"].median() for n, d in perm.items()})
print("\nTypical shuffle-to-shuffle noise (median std over the 14 features):")
print(perm_noise.round(3).to_string())
print("Differences smaller than about twice this noise are not reliable.")

# 10c. Do the models agree on the ranking? (Spearman rank correlation: 1 = same order)
agree = pd.DataFrame({**{f"Perm. {n}": perm_mean[n] for n in fitted},
                      "LR |coefficient|": coef.abs(), "RF impurity": rf_imp}).corr(method="spearman")
print("\nAgreement between the feature rankings (Spearman correlation):")
print(agree.round(2))
plot_perm_importance(perm)
perm_mean.round(4).to_csv(f"{OUT}/permutation_importance.csv")
built_in.round(4).to_csv(f"{OUT}/builtin_importance.csv")

# ===== STEP 11: Why do the models perform differently? Four experiments =====
step(11)
SC = SCALE
credit_raw = X["Credit_Score"] * SC["Credit_Score"][1] + SC["Credit_Score"][0]
income_raw = X["Annual_Income"] * SC["Annual_Income"][1] + SC["Annual_Income"][0]
loan_raw = X["Loan_Amount"] * SC["Loan_Amount"][1] + SC["Loan_Amount"][0]
ratio_raw = loan_raw / income_raw                              # loan-to-income: a standard lending ratio
r_tr = ratio_raw.loc[X_train.index]
X_fe = X.assign(Loan_to_Income=(ratio_raw - r_tr.mean()) / r_tr.std())   # scaled with training statistics
X_train_fe, X_test_fe = X_fe.loc[X_train.index], X_fe.loc[X_test.index]

# 11a. Feature ablation: the same tuned models on smaller and on engineered feature sets
financial = ["Credit_Score", "Annual_Income", "Loan_Amount"]
sets = {
    "Credit score only": ["Credit_Score"],
    "3 financial features": financial,
    "6 numeric features": ["Age", "Education", "Annual_Income", "Credit_Score", "Loan_Amount", "Loan_Term_Months"],
    "All 14 features": cols,
    "Credit score + loan-to-income": ["Credit_Score", "Loan_to_Income"],
    "All 14 + loan-to-income": cols + ["Loan_to_Income"],
}
ablation = pd.DataFrame({
    sname: {m: cross_val_score(clone(model), X_train_fe[fs], y_train, cv=rcv, scoring="roc_auc").mean()
            for m, model in fitted.items()} for sname, fs in sets.items()}).T
print("11a. Cross-validated ROC-AUC (15 folds) for each feature set:")
print(ablation.round(3))
plot_ablation(ablation)
ablation.round(4).to_csv(f"{OUT}/feature_ablation.csv")

# 11b. Learning curves: does more training data help?
lc, rows = {}, []
for name, model in fitted.items():
    n_tr, tr, va = learning_curve(clone(model), X_train, y_train, train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0], cv=skf,
                                  scoring="roc_auc", shuffle=True, random_state=42)
    lc[name] = (n_tr, tr, va)
    for n, a, b in zip(n_tr, tr.mean(axis=1), va.mean(axis=1)):
        rows.append({"Model": name, "Train_rows": int(n), "Train_AUC": a, "Validation_AUC": b, "Gap": a - b})
print("\n11b. Learning curves (5-fold CV at growing training sizes):")
print(pd.DataFrame(rows).set_index(["Model", "Train_rows"]).round(3))
plot_learning(lc)

# 11c. Are the differences between the models real? Corrected paired t-test on the 15 CV folds
#      (Nadeau-Bengio correction: folds overlap, so the plain t-test is too optimistic).
fold_auc = {m: cross_val_score(clone(model), X_train, y_train, cv=rcv, scoring="roc_auc")
            for m, model in fitted.items()}
pairs, rows = [("Logistic Regression", "KNN"), ("Logistic Regression", "Random Forest"),
               ("Random Forest", "KNN")], []
for a, b in pairs:
    d = fold_auc[a] - fold_auc[b]
    se = np.sqrt((1 / len(d) + 0.25) * d.var(ddof=1))          # test/train size ratio is 1/4 for 5 folds
    tstat = d.mean() / se
    rows.append({"Comparison": f"{a} - {b}", "Mean_AUC_diff": d.mean(), "t": tstat,
                 "p_value": 2 * t_dist.sf(abs(tstat), len(d) - 1), "Folds_A_better": int((d > 0).sum())})
paired = pd.DataFrame(rows).set_index("Comparison")
print("\n11c. Paired comparison of the tuned models (same 15 folds):")
print(paired.round(3))

# 11d. Noise ceiling. Because the dataset was generated in Week 1, the rule that produced the label is known.
#      Applying it to the cleaned data gives an approximate upper limit for ANY model.
unemp = X["Employment_Status_Unemployed"]
true_logit = (0.8 * (credit_raw - 600) / 50 + 0.5 * (income_raw - 650000) / 300000
              - 1.2 * (loan_raw / income_raw - 0.9) - 1.5 * unemp)
p_true = 1 / (1 + np.exp(-true_logit))
ceiling = {"AUC": roc_auc_score(y, p_true), "Accuracy": accuracy_score(y, (p_true >= 0.5).astype(int)),
           "AUC_test_rows": roc_auc_score(y_test, p_true.loc[y_test.index])}
print("\n11d. Approximate ceiling (true generating rule applied to the cleaned data):")
print(f"AUC on all 515 rows = {ceiling['AUC']:.3f}, accuracy = {ceiling['Accuracy']:.3f}, "
      f"AUC on the 103 test rows = {ceiling['AUC_test_rows']:.3f}")
print(f"Best model on the test rows ({best}): AUC = {test_metrics.loc[best, 'ROC_AUC']:.3f}")
print("(The label was drawn at random from these probabilities, so even the true rule cannot be perfect.)")

# ===== STEP 12: Error analysis - which applicants do the models get wrong? =====
step(12)
oof = {m: cross_val_predict(clone(model), X, y, cv=skf, method="predict_proba")[:, 1] for m, model in fitted.items()}
pred = {m: (p >= 0.5).astype(int) for m, p in oof.items()}      # out-of-fold: every row predicted by a model that never saw it
oof_tab = pd.DataFrame({m: {"AUC": roc_auc_score(y, oof[m]), "Accuracy": accuracy_score(y, pred[m]),
                            "Precision": precision_score(y, pred[m]), "Recall": recall_score(y, pred[m])}
                        for m in fitted}).T
print("Out-of-fold results on all 515 applicants (a larger sample than the 103-row test set):")
print(oof_tab.round(3))

wrong = pd.DataFrame({m: pred[m] != y.to_numpy() for m in fitted}, index=X.index)
n_wrong = wrong.sum(axis=1)
print("\nHow many of the 3 models get an applicant wrong:")
print(n_wrong.value_counts().sort_index().rename("Applicants").to_frame().assign(
    Share_pct=lambda d: (d["Applicants"] / len(X) * 100).round(1)))
print("\nPairwise disagreement (share of applicants where two models give different answers):")
names = list(fitted)
print(pd.DataFrame([[np.mean(pred[a] != pred[b]) for b in names] for a in names], index=names, columns=names).round(3))

emp = pd.Series(np.select([X["Employment_Status_Salaried"] == 1, X["Employment_Status_Self-Employed"] == 1,
                           unemp == 1], ["Salaried", "Self-Employed", "Unemployed"], "Business"), index=X.index)
credit_q = pd.qcut(credit_raw, 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
lti_q = pd.qcut(ratio_raw, 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
err = lambda g: wrong.groupby(g, observed=True).mean()
err_credit, err_lti, err_emp = err(credit_q), err(lti_q), err(emp)
print("\nError rate by credit-score quartile (Q1 = lowest scores):")
print(err_credit.round(3))
print("\nError rate by loan-to-income quartile (Q1 = smallest loan relative to income):")
print(err_lti.round(3))
print("\nError rate by employment status:")
print(err_emp.round(3))
print("\nApproval rate actually observed in each credit-score quartile (%):")
print((y.groupby(credit_q, observed=True).mean() * 100).round(1))
# Benchmark: inside a band, the best a model could do with NO extra information is to guess the majority outcome
best_guess = lambda g: y.groupby(g, observed=True).mean().apply(lambda p: min(p, 1 - p))
base_credit, base_lti = best_guess(credit_q), best_guess(lti_q)
print("\nError of the best constant guess inside each band (what a model must beat to add value):")
print(pd.DataFrame({"Credit_quartile": base_credit.to_numpy(), "LTI_quartile": base_lti.to_numpy()},
                   index=["Q1 lowest", "Q2", "Q3", "Q4 highest"]).round(3))

lr_pred = pred["Logistic Regression"]
fp_share = np.mean((lr_pred == 1) & (y.to_numpy() == 0))
fn_share = np.mean((lr_pred == 0) & (y.to_numpy() == 1))
print(f"\nLogistic Regression mistakes: wrongly approved {fp_share * 100:.1f}% of applicants, "
      f"wrongly rejected {fn_share * 100:.1f}%")
hard = n_wrong == 3
profile_hard = pd.DataFrame({"Wrong_for_all_3": [credit_raw[hard].mean(), ratio_raw[hard].mean(), income_raw[hard].mean(),
                                                 y[hard].mean() * 100],
                             "Correct_for_all_3": [credit_raw[n_wrong == 0].mean(), ratio_raw[n_wrong == 0].mean(),
                                                   income_raw[n_wrong == 0].mean(), y[n_wrong == 0].mean() * 100]},
                            index=["Credit score", "Loan-to-income", "Income (Rs.)", "Actually approved (%)"])
print(f"\nProfile of the {int(hard.sum())} applicants that all three models get wrong vs those all get right:")
print(profile_hard.round(2))
plot_errors(err_credit, err_lti, base_credit, base_lti)
oof_tab.round(4).to_csv(f"{OUT}/out_of_fold_results.csv")
# ===== END =====

step("end")
RESULTS = {
    "log": LOG, "best_k": best_k, "ari": float(ari), "best_model": best,
    "n95": n95, "evr": [float(v) for v in evr],
    "k_table": json.loads(k_table.reset_index().to_json(orient="records")),
    "profile": json.loads(profile.reset_index().rename(columns={"index": "Cluster"}).to_json(orient="records")),
    "cv": json.loads(cv_summary.reset_index().to_json(orient="records")),
    "tuned": json.loads(tuned_table.reset_index().to_json(orient="records")),
    "rs_best": float(rs.best_score_), "rf_grid_best": float(rf_search.best_score_),
    "test": json.loads(test_metrics.reset_index().to_json(orient="records")),
    "thr": json.loads(thr_table.reset_index().to_json(orient="records")),
    "cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    "built_in": json.loads(built_in.reset_index().rename(columns={"index": "Feature"}).to_json(orient="records")),
    "perm": json.loads(perm_mean.reset_index().rename(columns={"index": "Feature"}).to_json(orient="records")),
    "perm_noise": {k: float(v) for k, v in perm_noise.items()},
    "agree": json.loads(agree.reset_index().rename(columns={"index": "Name"}).to_json(orient="records")),
    "ablation": json.loads(ablation.reset_index().rename(columns={"index": "Set"}).to_json(orient="records")),
    "learning": {m: {"n": [int(v) for v in n_tr_], "train": [float(v) for v in tr_.mean(axis=1)],
                     "val": [float(v) for v in va_.mean(axis=1)]} for m, (n_tr_, tr_, va_) in lc.items()},
    "paired": json.loads(paired.reset_index().to_json(orient="records")),
    "ceiling": {k: float(v) for k, v in ceiling.items()},
    "oof": json.loads(oof_tab.reset_index().rename(columns={"index": "Model"}).to_json(orient="records")),
    "n_wrong": {int(k): int(v) for k, v in n_wrong.value_counts().sort_index().items()},
    "err_credit": json.loads(err_credit.reset_index().rename(columns={"index": "Band"}).to_json(orient="records")),
    "err_lti": json.loads(err_lti.reset_index().rename(columns={"index": "Band"}).to_json(orient="records")),
    "approval_credit": [float(v) for v in y.groupby(credit_q, observed=True).mean()],
    "approval_lti": [float(v) for v in y.groupby(lti_q, observed=True).mean()],
    "base_credit": [float(v) for v in base_credit], "base_lti": [float(v) for v in base_lti],
    "err_emp": json.loads(err_emp.reset_index().rename(columns={"index": "Band"}).to_json(orient="records")),
    "hard": json.loads(profile_hard.reset_index().rename(columns={"index": "Measure"}).to_json(orient="records")),
}
with open(f"{OUT}/log.json", "w") as f:
    json.dump(RESULTS, f, indent=2)
sys.stdout = sys.__stdout__
print("Finished. Outputs in", OUT)
