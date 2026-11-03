"""Builds report/Week3_Clustering_and_Model_Evaluation_Report.pdf from the real code, outputs and charts.
Run from the repository root after clustering_and_evaluation.py:   python3 src/build_report.py"""
import json
import re
import textwrap

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (CondPageBreak, Image, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

RES = "results"
PDF = "report/Week3_Clustering_and_Model_Evaluation_Report.pdf"
AUTHOR, DATE = "Shubh", "28 October 2026"

R = json.load(open(f"{RES}/log.json"))
LOG = R["log"]
cv = {r["Model"]: r for r in R["cv"]}
tn = {r["Model"]: r for r in R["tuned"]}
te = {r["Model"]: r for r in R["test"]}
thr = {r["Threshold"]: r for r in R["thr"]}
kt = {r["k"]: r for r in R["k_table"]}
prof = {r["Cluster"]: r for r in R["profile"]}
cm = R["cm"]
BEST = R["best_model"]
K = R["best_k"]
f3 = lambda v: f"{v:.3f}"

# ---------- real code of each step ----------
steps, titles, cur = {}, {}, None
for line in open("src/clustering_and_evaluation.py").read().splitlines():
    m = re.match(r"# ===== STEP (\d+): (.*) =====", line)
    if m:
        cur = int(m.group(1))
        titles[cur], steps[cur] = m.group(2), []
        continue
    if line.startswith("# ===== END") or line.startswith("# ===== SETUP"):
        cur = None
        continue
    if cur is not None and not line.startswith("step("):
        steps[cur].append(line)
CODE = {k: "\n".join(v).strip("\n") for k, v in steps.items()}


def wrap(text, width=118):
    out = []
    for ln in text.splitlines():
        if len(ln) <= width:
            out.append(ln)
        else:
            indent = len(ln) - len(ln.lstrip())
            out += textwrap.wrap(ln, width, subsequent_indent=" " * (indent + 4)) or [""]
    return "\n".join(out)


# ---------- styles ----------
INK, MUTED = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e")
BLUE = colors.HexColor("#2a78d6")
HDR, LIGHT, LINE = colors.HexColor("#22324a"), colors.HexColor("#f4f5f7"), colors.HexColor("#d5d7db")
ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=9.6, leading=14,
                      textColor=INK, spaceAfter=5)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=13, bulletIndent=2, spaceAfter=2.5)
h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=16, leading=20,
                    textColor=HDR, spaceBefore=4, spaceAfter=8)
h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=12, leading=16,
                    textColor=HDR, spaceBefore=8, spaceAfter=4)
label = ParagraphStyle("label", parent=body, fontName="Helvetica-Bold", fontSize=8, leading=10,
                       textColor=MUTED, spaceBefore=5, spaceAfter=2)
caption = ParagraphStyle("caption", parent=body, fontSize=8, leading=10.5, textColor=MUTED, spaceAfter=8)
cell = ParagraphStyle("cell", parent=body, fontSize=8.4, leading=11, spaceAfter=0)
cell_h = ParagraphStyle("cell_h", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
code_style = ParagraphStyle("code", fontName="Courier", fontSize=6.7, leading=8.3, textColor=INK,
                            backColor=LIGHT, borderPadding=(4, 4, 4, 4), leftIndent=4, rightIndent=4,
                            spaceBefore=3, spaceAfter=7)
out_style = ParagraphStyle("out", parent=code_style, backColor=colors.HexColor("#fbf6f2"))
title_style = ParagraphStyle("title", parent=h1, fontSize=27, leading=33, spaceAfter=6)
sub_style = ParagraphStyle("sub", parent=body, fontSize=13, leading=18, textColor=MUTED, spaceAfter=18)
W = A4[0] - 36 * mm


def P(t, s=body):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(t, bullet, bulletText="•") for t in items]


def table(rows, widths):
    d = [[Paragraph(str(c), cell_h if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(d, colWidths=[w * W for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HDR), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, LINE), ("BOX", (0, 0), (-1, -1), 0.3, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return [t, Spacer(1, 8)]


def image(name, cap, width=0.92):
    path = f"{RES}/charts/{name}.png"
    iw, ih = ImageReader(path).getSize()
    w = W * width
    return [Image(path, width=w, height=w * ih / iw), P(cap, caption)]


def callout(text):
    t = Table([[Paragraph(text, cell)]], colWidths=[W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eaf2fc")),
        ("LINEBEFORE", (0, 0), (0, -1), 3, BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return [t, Spacer(1, 6)]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "Week 3 Task - Unsupervised Learning and Model Evaluation")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


# ---------- tables built from real results ----------
c0, c1 = prof[0], prof[1]
prof_rows = [["Cluster", "Size", "Avg age", "Education (0-3)", "Avg income (Rs.)", "Credit score",
              "Avg loan (Rs.)", "Approval rate"]]
for c in (0, 1):
    p = prof[c]
    prof_rows.append([f"Cluster {c}", int(p["Size"]), f"{p['Age']:.1f}", f"{p['Education']:.2f}",
                      f"{p['Annual_Income']:,.0f}", f"{p['Credit_Score']:.0f}",
                      f"{p['Loan_Amount']:,.0f}", f"{p['Approval_%']:.1f}%"])

cv_rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Train AUC"]]
for n in cv:
    r = cv[n]
    cv_rows.append([n] + [f"{r[f'{s}_mean']:.3f} +/- {r[f'{s}_std']:.3f}" for s in
                          ("accuracy", "precision", "recall", "f1", "roc_auc")] + [f3(r["train_auc"])])
test_rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
for n in te:
    r = te[n]
    test_rows.append([n, f3(r["Accuracy"]), f3(r["Precision"]), f3(r["Recall"]), f3(r["F1"]), f3(r["ROC_AUC"])])
tune_rows = [["Model", "Search", "Combinations", "Model fits", "Best parameters", "CV AUC"]]
for n in tn:
    r = tn[n]
    tune_rows.append([n, r["Search"], r["Combinations"], r["Model_fits"], r["Best_parameters"], f3(r["CV_AUC"])])
thr_rows = [["Threshold", "Precision", "Recall", "F1", "Accuracy"]]
for t in sorted(thr):
    r = thr[t]
    thr_rows.append([f"{t:.1f}", f3(r["Precision"]), f3(r["Recall"]), f3(r["F1"]), f3(r["Accuracy"])])

base, lrt = te["Baseline (always approve)"], te[BEST]

# ---------- data for Part C (steps 10-12) ----------
M3 = ["Logistic Regression", "KNN", "Random Forest"]
perm = {r["Feature"]: r for r in R["perm"]}
bi = {r["Feature"]: r for r in R["built_in"]}
abl = {r["Set"]: r for r in R["ablation"]}
lc = R["learning"]
pair = {r["Comparison"]: r for r in R["paired"]}
ceil = R["ceiling"]
oof = {r["Model"]: r for r in R["oof"]}
nw = R["n_wrong"]
ec = {r["Credit_Score"]: r for r in R["err_credit"]}
el = {r["Loan_Amount"] if "Loan_Amount" in r else r.get("Band"): r for r in R["err_lti"]}
hard = {r["Measure"]: r for r in R["hard"]}
noise = R["perm_noise"]
QL = ["Q1 lowest", "Q2", "Q3", "Q4 highest"]
err_lti_rows = R["err_lti"]
err_credit_rows = R["err_credit"]
agree = {r["Name"]: r for r in R["agree"]}
f2 = lambda v: f"{v:.2f}"
pct = lambda v: f"{v * 100:.1f}%"

top_feats = sorted(perm, key=lambda k: -perm[k]["Average"])[:8]
perm_rows = [["Feature", "Logistic Regression", "KNN", "Random Forest", "Average"]]
for k in top_feats:
    perm_rows.append([k] + [f3(perm[k][m]) for m in M3] + [f3(perm[k]["Average"])])
built_rows = [["Feature", "LR coefficient", "LR rank", "RF importance", "RF rank"]]
for k in sorted(bi, key=lambda k: bi[k]["RF_rank"])[:8]:
    built_rows.append([k, f"{bi[k]['LR_coefficient']:+.3f}", bi[k]["LR_rank"], f3(bi[k]["RF_importance"]),
                       bi[k]["RF_rank"]])
agree_names = ["Perm. Logistic Regression", "Perm. KNN", "Perm. Random Forest", "LR |coefficient|", "RF impurity"]
agree_rows = [[""] + [n.replace("Perm. ", "Perm: ").replace("Logistic Regression", "LR").replace("Random Forest", "RF")
                      for n in agree_names]]
for a in agree_names:
    agree_rows.append([a.replace("Perm. ", "Perm: ").replace("Logistic Regression", "LR")
                       .replace("Random Forest", "RF")] + [f2(agree[a][b]) for b in agree_names])
abl_rows = [["Feature set"] + M3]
for k in abl:
    abl_rows.append([k] + [f3(abl[k][m]) for m in M3])
lc_rows = [["Model", "Score"] + [str(n) + " rows" for n in lc["Logistic Regression"]["n"]]]
for m in M3:
    lc_rows.append([m, "Training AUC"] + [f3(v) for v in lc[m]["train"]])
    lc_rows.append(["", "Validation AUC"] + [f3(v) for v in lc[m]["val"]])
pair_rows = [["Comparison", "Mean AUC difference", "Folds where first is better", "Corrected t", "p-value"]]
for k, r in pair.items():
    pair_rows.append([k, f"{r['Mean_AUC_diff']:+.3f}", f"{r['Folds_A_better']} of 15", f2(r["t"]), f2(r["p_value"])])
oof_rows = [["Model", "AUC", "Accuracy", "Precision", "Recall"]]
for m in M3:
    r = oof[m]
    oof_rows.append([m, f3(r["AUC"]), f3(r["Accuracy"]), f3(r["Precision"]), f3(r["Recall"])])
err_rows = [["Credit-score quartile", "Actually approved", "Best constant guess", "LR error", "KNN error", "RF error"]]
for i, q in enumerate(QL):
    r = err_credit_rows[i]
    err_rows.append([q, pct(R["approval_credit"][i]), pct(R["base_credit"][i]), pct(r["Logistic Regression"]),
                     pct(r["KNN"]), pct(r["Random Forest"])])
err2_rows = [["Loan-to-income quartile", "Actually approved", "Best constant guess", "LR error", "KNN error",
              "RF error"]]
for i, q in enumerate(QL):
    r = err_lti_rows[i]
    err2_rows.append([q, pct(R["approval_lti"][i]), pct(R["base_lti"][i]), pct(r["Logistic Regression"]),
                      pct(r["KNN"]), pct(r["Random Forest"])])
errw = {r["Credit_Score"] if "Credit_Score" in r else r["Band"]: r for r in R["err_emp"]}
n_all = sum(nw.values())
share = lambda k: nw[str(k)] / n_all * 100
lr_oof, rf_oof, kn_oof = oof["Logistic Regression"], oof["Random Forest"], oof["KNN"]

STEP_TEXT = {
    1: ("Clustering is <b>unsupervised</b>: there is no target column to predict, the algorithm looks for "
        "natural groups on its own. The label <i>Approved</i> is therefore <b>not used</b>. Only the six "
        "columns that describe an applicant are used (age, education level, income, credit score, loan amount, "
        "loan term). The 0/1 dummy columns are left out because they would dominate the distance calculation. "
        "Clustering is distance-based, so all features are first put on the same scale with "
        "<b>StandardScaler</b> (mean 0, standard deviation 1).",
        "515 applicants described by 6 scaled features."),
    2: ("<b>Dimensionality reduction</b> compresses many features into a few new ones, called principal "
        "components, that keep as much information (variance) as possible. <b>PCA</b> finds the directions in "
        "which the data varies most. It is used here to see how many independent patterns exist and to draw the "
        "data in 2-D. The loadings show which original features build each component.",
        f"PC1 keeps {R['evr'][0] * 100:.0f}% and PC2 {R['evr'][1] * 100:.0f}% of the variance, so a 2-D plot shows "
        f"about {(R['evr'][0] + R['evr'][1]) * 100:.0f}% of the information. {R['n95']} of 6 components are "
        "needed for 95%, which means the features carry mostly separate information and the data cannot be "
        "squeezed much further. PC1 is an overall 'size' axis: income, age and loan amount rise together."),
    3: ("<b>K-Means</b> places k centres, assigns every applicant to the nearest centre, moves each centre to the "
        "average of its members, and repeats until nothing changes. k must be chosen. Two guides are used: the "
        "<b>elbow method</b> (inertia, the total distance to the centres, always falls as k grows; look for a "
        "bend) and the <b>silhouette score</b> (from -1 to 1: how much closer an applicant is to its own cluster "
        "than to the nearest other one; higher is better).",
        f"The silhouette score is highest at k = {K} ({kt[K]['Silhouette']:.3f}) and the elbow curve has no "
        f"clear bend, so k = {K} is chosen. <b>Important:</b> a silhouette of about 0.23 is low. Values above "
        "0.5 mean clearly separated groups; here the applicants form one continuous cloud, and the clusters "
        "split it into two halves rather than finding natural islands."),
    4: ("<b>Hierarchical clustering</b> starts with every applicant as its own cluster and repeatedly merges the "
        "two closest clusters, building a tree called a <b>dendrogram</b>. Cutting the tree at a chosen height "
        "gives the clusters. The <b>linkage</b> rule decides how distance between clusters is measured; "
        "<b>Ward</b> merges the pair that increases the within-cluster variance least. The cluster numbers of "
        "the two methods are matched with the Hungarian algorithm, and the <b>Adjusted Rand Index (ARI)</b> "
        "measures how much they agree (1 = identical, 0 = random).",
        f"Ward linkage gives balanced clusters (198+ applicants each). The single and average linkages reach a "
        f"higher silhouette score only because they create tiny or one-person clusters (smallest cluster: 1 and 5 "
        f"applicants), which is useless. The ARI between K-Means and Ward is {R['ari']:.3f}, a <b>moderate</b> "
        "agreement: the two methods draw the boundary in somewhat different places, another sign that the "
        "groups are not sharply separated."),
    5: ("A cluster is only useful if it can be described. The average of every feature is computed per cluster and "
        "converted back to real units with the Week 1 means and standard deviations. The approval rate per "
        "cluster is an <b>independent check</b>, because the label was never given to the algorithm.",
        f"<b>Cluster 0</b> ({int(c0['Size'])} applicants) are older, more educated, higher-income applicants with "
        f"larger loans (average age {c0['Age']:.0f}, income Rs. {c0['Annual_Income']:,.0f}). "
        f"<b>Cluster 1</b> ({int(c1['Size'])}) are younger, lower-income applicants with smaller loans "
        f"(age {c1['Age']:.0f}, income Rs. {c1['Annual_Income']:,.0f}). Approval is "
        f"{c0['Approval_%']:.0f}% in cluster 0 versus {c1['Approval_%']:.0f}% in cluster 1, so the grouping "
        "carries real information about approval even though approval was not used."),
    6: ("A single train/test split can be lucky or unlucky. <b>Cross-validation</b> repeats the test many times: "
        "the training data is cut into 5 folds, each fold is used once for validation, and the whole process is "
        "repeated 3 times with different cuts (15 rounds). The mean shows the typical score and the "
        "<b>standard deviation</b> shows how stable it is. A <b>baseline</b> that always predicts 'approved' is "
        "included, because a model is only useful if it beats the trivial answer. The gap between the training "
        "score and the validation score reveals <b>overfitting</b>.",
        f"The always-approve baseline already reaches accuracy {f3(cv['Baseline (always approve)']['accuracy_mean'])} "
        f"and an F1 of {f3(cv['Baseline (always approve)']['f1_mean'])}, because 61% of applicants are approved. "
        f"Logistic Regression matches the baseline F1 ({f3(cv['Logistic Regression']['f1_mean'])}) and clearly "
        f"beats it on accuracy and ROC-AUC ({f3(cv['Logistic Regression']['roc_auc_mean'])} vs 0.500). "
        "<b>ROC-AUC is the metric that shows real skill here</b>, because accuracy and F1 flatter a model on an "
        "unbalanced label. The Random Forest memorises the training folds (training AUC 1.000 vs validation "
        f"{f3(cv['Random Forest']['roc_auc_mean'])}): a clear case of overfitting. The spread of about 0.06 AUC "
        "between folds shows that differences smaller than that are noise."),
    7: ("<b>Hyperparameters</b> are settings chosen before training (number of neighbours, tree depth, "
        "regularisation strength). <b>GridSearchCV</b> tries every combination with 5-fold cross-validation and "
        "keeps the best, scored by ROC-AUC on the <b>training data only</b>. <b>RandomizedSearchCV</b> tries a "
        "fixed number of random combinations from a much wider range, which is cheaper when there are many "
        "settings. A <b>validation curve</b> varies one setting and plots the training score against the "
        "validation score.",
        f"Best settings: Logistic Regression C = 0.1, KNN k = 31 with distance weighting, Random Forest depth 4. "
        f"RandomizedSearchCV found a slightly higher forest score ({f3(R['rs_best'])} vs "
        f"{f3(R['rf_grid_best'])}) with 20% fewer model fits, but the gap is far below the fold-to-fold spread, "
        "so it is not a real difference. The validation curve shows that deeper forests keep improving on the "
        "training data (to 1.00) while the validation score stays flat at about 0.72: more complexity brings no "
        "benefit, only overfitting."),
    8: ("The tuned models are scored once on the untouched test set (103 applicants). The <b>confusion matrix</b> "
        "counts the four outcomes: true positives (TP, approved and predicted approved), true negatives (TN), "
        "false positives (FP, rejected but predicted approved) and false negatives (FN, approved but predicted "
        "rejected). <b>Precision</b> = TP / (TP + FP) answers 'of those predicted approved, how many really "
        "were?'. <b>Recall</b> = TP / (TP + FN) answers 'of those really approved, how many did we find?'. "
        "<b>F1</b> is their harmonic mean. <b>ROC-AUC</b> measures how well the model ranks approved above "
        "rejected applicants at every threshold. The best model is chosen by cross-validated AUC, not by the "
        "test score.",
        f"<b>Best model: {BEST}</b> (accuracy {f3(lrt['Accuracy'])}, precision {f3(lrt['Precision'])}, recall "
        f"{f3(lrt['Recall'])}, F1 {f3(lrt['F1'])}, ROC-AUC {f3(lrt['ROC_AUC'])}). Its confusion matrix has "
        f"TP = {cm['tp']}, TN = {cm['tn']}, FP = {cm['fp']}, FN = {cm['fn']}. It finds most approved applicants "
        f"(recall 0.84) but is weak at spotting rejections: only {cm['tn']} of 40 rejected applicants were "
        "caught (specificity 0.525). The baseline scores accuracy "
        f"{f3(base['Accuracy'])} and AUC 0.500, so the models add real but modest skill (+0.11 accuracy, "
        "+0.26 AUC). The three tuned models are statistically indistinguishable on 103 test rows."),
    9: ("A classifier outputs a probability, and the <b>decision threshold</b> turns it into a yes/no answer. The "
        "default is 0.5, but it can be moved depending on which mistake is more costly. A higher threshold "
        "approves fewer applicants: precision rises and recall falls. A lower threshold does the opposite.",
        f"At threshold 0.7 precision rises to {f3(thr[0.7]['Precision'])} but recall collapses to "
        f"{f3(thr[0.7]['Recall'])}. At 0.4 recall reaches {f3(thr[0.4]['Recall'])} and F1 peaks at "
        f"{f3(thr[0.4]['F1'])}. For a lender who mostly fears bad loans (false positives) a threshold around "
        "0.6 is safer; for a lender who does not want to lose good customers a threshold around 0.4 is better. "
        "The right value is a business decision, not a statistical one."),
    10: ("Feature importance answers the question 'which columns does a model actually use?'. Three views are "
         "compared. <b>Logistic Regression coefficients</b> show the direction and size of each feature's effect "
         "(numeric columns are scaled, so a coefficient is the change in log-odds per standard deviation; the 0/1 "
         "dummy columns are not scaled, so their coefficients are not directly comparable). <b>Random Forest "
         "impurity importance</b> adds up how much each feature reduces impurity across all trees; it is known to "
         "favour columns with many distinct values. <b>Permutation importance</b> shuffles one column of the test "
         "set and measures how much the ROC-AUC drops. It works the same way for every model, including KNN, "
         "which has no built-in importance, and it measures what the model really relies on. The shuffle is "
         "repeated 30 times and its spread (noise) shows which differences are real. <b>Spearman correlation</b> "
         "measures how much two rankings agree (1 = same order).",
         f"<b>Every model and every method puts credit score first.</b> Shuffling it drops the test AUC by "
         f"{f3(perm['Credit_Score']['Logistic Regression'])} (Logistic Regression), "
         f"{f3(perm['Credit_Score']['KNN'])} (KNN) and {f3(perm['Credit_Score']['Random Forest'])} (Random "
         "Forest), which takes each model from about 0.75 down to 0.53-0.57, close to random guessing. "
         "<b>Below credit score the models disagree.</b> Logistic Regression uses Annual_Income "
         f"({f3(perm['Annual_Income']['Logistic Regression'])}) and Loan_Amount "
         f"({f3(perm['Loan_Amount']['Logistic Regression'])}) together, with opposite-sign coefficients "
         f"({bi['Loan_Amount']['LR_coefficient']:+.2f} and {bi['Annual_Income']['LR_coefficient']:+.2f}): a big "
         "loan relative to income lowers approval. KNN and Random Forest rely on Age instead "
         f"({f3(perm['Age']['KNN'])} and {f3(perm['Age']['Random Forest'])}), probably as a stand-in for income "
         "(age and income are correlated, r = 0.74 in Week 1). The shuffle noise is only about "
         f"{noise['Logistic Regression']:.3f} (Logistic Regression), {noise['KNN']:.3f} (KNN) and "
         f"{noise['Random Forest']:.3f} (Random Forest), so apart from credit score most differences are small. "
         f"Rank agreement between models is low (Spearman {f2(agree['Perm. Logistic Regression']['Perm. KNN'])} "
         f"LR-KNN, {f2(agree['Perm. Logistic Regression']['Perm. Random Forest'])} LR-RF, "
         f"{f2(agree['Perm. KNN']['Perm. Random Forest'])} KNN-RF). <b>Two warnings.</b> Impurity importance "
         f"ranks Loan_Term_Months ({f3(bi['Loan_Term_Months']['RF_importance'])}) and Education "
         f"({f3(bi['Education']['RF_importance'])}) 5th and 6th although shuffling them does not lower the AUC, "
         "so it overstates them. And the 'Unemployed' dummy has the second-largest regression coefficient "
         f"({bi['Employment_Status_Unemployed']['LR_coefficient']:+.2f}) but almost no effect on AUC "
         f"({f3(perm['Employment_Status_Unemployed']['Logistic Regression'])}), because it describes only about "
         "5% of applicants: a large coefficient is not the same as an important feature."),
    11: ("Four experiments look for the reasons behind the ranking of the models. <b>(a) Feature ablation:</b> "
         "the same tuned models are cross-validated on smaller feature sets, and on a set with an "
         "<b>engineered feature</b>, loan-to-income (loan amount divided by income, a standard lending ratio), "
         "to see which model is hurt by useless columns or helped by better inputs. <b>(b) Learning curves</b> "
         "train on growing parts of the training data: a large gap between training and validation score means "
         "high variance (the model memorises), and a validation score that is still rising means more data would "
         "help. <b>(c) A corrected paired t-test</b> on the same 15 folds asks whether the differences between "
         "the models are real; the Nadeau-Bengio correction is used because the folds overlap and a plain "
         "t-test would be too optimistic. <b>(d) Noise ceiling:</b> because the dataset was generated in Week 1, "
         "the rule that produced the approval label is known. Applying it to the cleaned data gives an "
         "approximate upper limit for any model. This is only possible for synthetic data.",
         f"<b>(a)</b> Logistic Regression is steady "
         f"({min(abl[k]['Logistic Regression'] for k in abl):.3f}-{max(abl[k]['Logistic Regression'] for k in abl):.3f} "
         "whatever the features). KNN is hurt most by extra columns "
         f"({f3(abl['3 financial features']['KNN'])} with 3 financial features vs "
         f"{f3(abl['All 14 features']['KNN'])} with all 14) and Random Forest also gains from fewer columns "
         f"({f3(abl['3 financial features']['Random Forest'])} vs {f3(abl['All 14 features']['Random Forest'])}). "
         "Adding the loan-to-income ratio to credit score lifts every model above credit score alone "
         f"(Logistic Regression {abl['Credit score + loan-to-income']['Logistic Regression'] - abl['Credit score only']['Logistic Regression']:+.3f}, "
         f"KNN {abl['Credit score + loan-to-income']['KNN'] - abl['Credit score only']['KNN']:+.3f}, "
         f"Random Forest {abl['Credit score + loan-to-income']['Random Forest'] - abl['Credit score only']['Random Forest']:+.3f}), "
         "but these gains are smaller than the fold-to-fold spread. "
         f"<b>(b)</b> For Logistic Regression the training and validation scores meet (gap "
         f"{lc['Logistic Regression']['train'][-1] - lc['Logistic Regression']['val'][-1]:.2f} at full size): "
         "the model is stable and more data will not help. KNN's training AUC is 1.00 at every size, because with "
         "distance weighting each training point is its own nearest neighbour, so its gap is "
         f"{lc['KNN']['train'][-1] - lc['KNN']['val'][-1]:.2f}. The Random Forest gap shrinks from "
         f"{lc['Random Forest']['train'][0] - lc['Random Forest']['val'][0]:.2f} to "
         f"{lc['Random Forest']['train'][-1] - lc['Random Forest']['val'][-1]:.2f} and its validation AUC is still "
         f"rising ({f3(lc['Random Forest']['val'][0])} to {f3(lc['Random Forest']['val'][-1])}): it is the model "
         "that would gain most from more data. "
         f"<b>(c)</b> Logistic Regression beats Random Forest by "
         f"{pair['Logistic Regression - Random Forest']['Mean_AUC_diff']:+.3f} AUC and KNN by "
         f"{pair['Logistic Regression - KNN']['Mean_AUC_diff']:+.3f} on average (better in 11 of 15 folds), but "
         f"the corrected p-values are {pair['Logistic Regression - Random Forest']['p_value']:.2f} and "
         f"{pair['Logistic Regression - KNN']['p_value']:.2f}: <b>the ranking of the models is not statistically "
         "established</b>. "
         f"<b>(d)</b> The true rule gives an AUC of {f3(ceil['AUC'])} (accuracy {f3(ceil['Accuracy'])}) on all 515 "
         f"rows and {f3(ceil['AUC_test_rows'])} on the test rows. The best model reaches {f3(lr_oof['AUC'])} "
         f"(accuracy {f3(lr_oof['Accuracy'])}) out-of-fold and {f3(lrt['ROC_AUC'])} on the test rows, so it is "
         "already at the approximate ceiling (the AUC estimate itself has an uncertainty of about 0.02)."),
    12: ("Averages hide where a model fails. To use all 515 applicants, <b>out-of-fold predictions</b> are made: "
         "each applicant is predicted by a model that never saw them (5-fold). Errors are then grouped by "
         "credit-score band, loan-to-income band and employment status. A benchmark is added: inside any band, "
         "the <b>best constant guess</b> (always predict the more common outcome of that band) makes errors "
         "equal to the smaller share. A model only adds value if it beats this benchmark. Finally, the "
         "applicants that all three models get wrong are compared with those that all get right.",
         f"Out-of-fold AUC is {f3(lr_oof['AUC'])} (Logistic Regression), {f3(rf_oof['AUC'])} (Random Forest) "
         f"and {f3(kn_oof['AUC'])} (KNN). {share(0):.1f}% of applicants are classified correctly by all three "
         f"models, {share(3):.1f}% wrongly by all three, and only {share(1) + share(2):.1f}% split the models: "
         "the models make mostly the <b>same</b> mistakes. Errors concentrate where the credit score is low or "
         f"middling (Logistic Regression error {pct(err_credit_rows[0]['Logistic Regression'])} and "
         f"{pct(err_credit_rows[1]['Logistic Regression'])} in the two lowest quartiles vs "
         f"{pct(err_credit_rows[3]['Logistic Regression'])} in the highest), but in those bands the model is no "
         f"better than the best constant guess ({pct(R['base_credit'][0])} and {pct(R['base_credit'][1])}; "
         "it is worse in the lowest band and less than one point better in the second). The model's skill comes from separating high from low credit "
         "scores; inside a band the other features add no usable signal. "
         f"The {nw['3']} applicants that all three models get wrong were approved only "
         f"{hard['Actually approved (%)']['Wrong_for_all_3']:.0f}% of the time, against "
         f"{hard['Actually approved (%)']['Correct_for_all_3']:.0f}% for those all get right, although their "
         f"profiles are similar (credit score {hard['Credit score']['Wrong_for_all_3']:.0f} vs "
         f"{hard['Credit score']['Correct_for_all_3']:.0f}; loan-to-income "
         f"{hard['Loan-to-income']['Wrong_for_all_3']:.2f} vs {hard['Loan-to-income']['Correct_for_all_3']:.2f}). "
         "Their outcome contradicts their profile, which no better model could predict. KNN is weakest in the "
         f"sparse low-credit bands ({pct(err_credit_rows[0]['KNN'])} and {pct(err_credit_rows[1]['KNN'])} errors "
         f"vs {pct(err_credit_rows[0]['Logistic Regression'])} and "
         f"{pct(err_credit_rows[1]['Logistic Regression'])} for Logistic Regression) and for the small group of "
         f"unemployed applicants ({pct(errw['Unemployed']['KNN'])} vs {pct(errw['Unemployed']['Logistic Regression'])})."),
}
FIGS = {
    2: [("01_pca_variance", "Figure 1 - Variance kept by each principal component (bars) and in total (line).", 0.7)],
    3: [("02_kmeans_selection", "Figure 2 - Elbow curve (left) and silhouette score (right) for k = 2 to 8.", 0.92)],
    4: [("03_dendrogram", "Figure 3 - Dendrogram. The dashed line shows where the tree is cut to get the "
                          "clusters.", 0.92),
        ("04_clusters_pca", "Figure 4 - The two clusterings drawn on the first two principal components. "
                            "The clusters touch and overlap: there is no empty gap between them.", 0.95)],
    5: [("05_cluster_profiles", "Figure 5 - How each cluster differs from the average applicant, per feature.", 0.7)],
    6: [("06_cross_validation", "Figure 6 - Cross-validated scores. The lines show the standard deviation over "
                                "15 folds.", 0.88)],
    7: [("07_validation_curve", "Figure 7 - Training score keeps rising with depth; validation score stays flat.", 0.66)],
    8: [("08_confusion_matrices", "Figure 8 - Confusion matrices of the tuned models (rows = actual, "
                                  "columns = predicted).", 0.92),
        ("09_roc_curves", "Figure 9 - ROC curves: the closer to the top-left corner, the better.", 0.55)],
    9: [("10_threshold_tradeoff", "Figure 10 - Precision, recall and F1 as the decision threshold moves.", 0.7)],
    10: [("11_feature_importance_models", "Figure 11 - Permutation importance of the eight most important features "
                                           "for each model. The lines show the spread over 30 shuffles.", 0.88)],
    11: [("12_feature_ablation", "Figure 12 - Cross-validated ROC-AUC of each model on six feature sets.", 0.9),
         ("13_learning_curves", "Figure 13 - Learning curves: training score (dashed) and validation score "
                                "(solid) at growing training sizes.", 0.95)],
    12: [("14_error_analysis", "Figure 14 - Error rate of each model inside credit-score and loan-to-income "
                               "bands. The dotted line is the error of the best constant guess in the band.", 0.95)],
}
EXTRA = {
    5: ("Cluster profiles (K-Means)", prof_rows, [0.11, 0.07, 0.09, 0.13, 0.16, 0.11, 0.16, 0.17]),
    6: ("Cross-validation results (mean +/- standard deviation over 15 folds)", cv_rows,
        [0.20, 0.15, 0.15, 0.15, 0.15, 0.15, 0.05 + 0.0]),
    7: ("Tuning results", tune_rows, [0.17, 0.08, 0.12, 0.09, 0.42, 0.12]),
    8: ("Test-set results of the tuned models", test_rows, [0.31, 0.14, 0.14, 0.14, 0.13, 0.14]),
    9: ("Effect of the decision threshold", thr_rows, [0.2, 0.2, 0.2, 0.2, 0.2]),
}
EXTRA[6] = (EXTRA[6][0], EXTRA[6][1], [0.19, 0.15, 0.15, 0.14, 0.14, 0.14, 0.09])
EXTRA = {k: [v] for k, v in EXTRA.items()}                 # every step may now have several tables
EXTRA[10] = [("Permutation importance (drop in test ROC-AUC) of the 8 most important features", perm_rows,
              [0.34, 0.17, 0.15, 0.17, 0.17]),
             ("Built-in importance (Logistic Regression coefficient, Random Forest impurity)", built_rows,
              [0.36, 0.16, 0.12, 0.20, 0.16]),
             ("Agreement between the importance rankings (Spearman correlation)", agree_rows,
              [0.22, 0.16, 0.14, 0.18, 0.15, 0.15])]
EXTRA[11] = [("Cross-validated ROC-AUC for each feature set (11a)", abl_rows, [0.40, 0.20, 0.20, 0.20]),
             ("Learning curves: ROC-AUC at growing training sizes (11b)", lc_rows,
              [0.20, 0.16, 0.128, 0.128, 0.128, 0.128, 0.128]),
             ("Corrected paired comparison on the same 15 folds (11c)", pair_rows, [0.34, 0.16, 0.18, 0.14, 0.18])]
EXTRA[12] = [("Out-of-fold results on all 515 applicants", oof_rows, [0.30, 0.175, 0.175, 0.175, 0.175]),
             ("Error rate by credit-score quartile", err_rows, [0.20, 0.16, 0.18, 0.15, 0.15, 0.16]),
             ("Error rate by loan-to-income quartile", err2_rows, [0.20, 0.16, 0.18, 0.15, 0.15, 0.16])]

st = []
# ---------- cover ----------
st += [Spacer(1, 35 * mm), P("Unsupervised Learning<br/>and Model Evaluation", title_style),
       P("Week 3 Task Report - Clustering and Model Evaluation Report", sub_style)]
st += table([
    ["Task", "Perform clustering on a dataset and prepare a model evaluation report"],
    ["Prepared by", AUTHOR], ["Date", DATE],
    ["Tools", "Python 3, Scikit-learn, SciPy, Pandas, NumPy, Matplotlib"],
    ["Data", "Cleaned loan applicant dataset from Week 1 (515 rows, 15 columns)"],
    ["Part A", "Unsupervised learning: PCA, K-Means and Hierarchical clustering, cluster interpretation"],
    ["Part B", "Model evaluation: cross-validation, confusion matrix, precision, recall, F1, ROC-AUC, "
               "hyperparameter tuning, decision threshold"],
    ["Part C", "Model analysis: feature importance across models, why the models perform differently "
               "(ablation, learning curves, significance test, noise ceiling), error analysis"],
    ["Code", "src/clustering_and_evaluation.py (full code and results in the GitHub repository)"],
], [0.2, 0.8])
st += [Spacer(1, 6), P("<b>Contents</b>", h2)]
st += bullets(["1. Objective and overview", "2. Key concepts", "3. Step-by-step documentation (Steps 1 to 12)",
               "4. Discussion: why the models perform differently and how feature importance varies",
               "5. Summary of findings", "6. Limitations and next steps", "Appendix: repository and how to run"])
st.append(PageBreak())

# ---------- 1 objective ----------
st += [P("1. Objective and overview", h1),
       P("This report covers the Week 3 objectives:")]
st += bullets([
    "Understand clustering algorithms (K-Means and Hierarchical Clustering) and dimensionality reduction (PCA).",
    "Learn cross-validation, the confusion matrix, precision, recall, F1-score, ROC-AUC and hyperparameter tuning.",
    "<b>Deliverable:</b> perform clustering on a dataset and prepare a model evaluation report.",
])
st.append(Spacer(1, 3))
st.append(P("The work is split into three parts using the cleaned loan applicant dataset from Weeks 1 and 2. "
            "<b>Part A</b> (Steps 1-5) clusters the applicants into groups without using the approval label "
            "and interprets the groups. <b>Part B</b> (Steps 6-9) evaluates classification models that predict "
            "loan approval, with a focus on <i>how</i> a model should be judged: a baseline for comparison, "
            "cross-validation for stable scores, tuning, and the choice of decision threshold. <b>Part C</b> "
            "(Steps 10-12 and Section 4) explains <i>why</i> the models perform as they do: which features each "
            "model relies on, what separates the best model from the weakest, whether the differences are real, "
            "how close the models are to the best score the data allows, and which applicants are misclassified."))

# ---------- 2 concepts ----------
st += [P("2. Key concepts", h1)]
st += table([
    ["Concept", "In one sentence"],
    ["Unsupervised learning", "Finds structure in data that has no labels (no correct answer to learn from)."],
    ["K-Means", "Splits data into k groups around k centres that are moved until the groups are stable. "
                "Needs k in advance and assumes round, similarly sized groups."],
    ["Hierarchical clustering", "Builds a tree by merging the closest clusters step by step; the tree can be cut "
                                "at any height. No need to choose k in advance."],
    ["PCA", "Rotates the data to new axes (components) ordered by how much variation they capture, so a few "
            "components can summarise many features."],
    ["Silhouette score", "-1 to 1: how well each point fits its own cluster versus the next nearest. Above 0.5 "
                         "is clear structure; below 0.25 is weak."],
    ["Cross-validation", "Repeats training and testing on different splits of the data to get a stable score "
                         "and its spread."],
    ["Confusion matrix", "Table of true positives, true negatives, false positives and false negatives."],
    ["Precision / Recall / F1", "Precision: how many predicted positives are right. Recall: how many real "
                                "positives are found. F1: the balance of both."],
    ["ROC-AUC", "How well the model ranks positives above negatives over all thresholds: 0.5 is guessing, 1.0 "
                "is perfect."],
    ["Hyperparameter tuning", "Searching for the best model settings with cross-validation (grid or random "
                              "search) without touching the test set."],
    ["Permutation importance", "Shuffles one feature and measures how much the model's score drops; works for any "
                               "model and shows what the model really relies on."],
    ["Learning curve", "Training and validation score at growing training sizes; shows whether a model memorises "
                       "(large gap) and whether more data would help (validation still rising)."],
    ["Feature ablation", "Re-running a model with fewer (or engineered) features to see which features help or hurt."],
    ["Noise ceiling", "The best score any model can reach when the outcome is partly random; a model cannot "
                      "be blamed for missing what is not predictable."],
], [0.24, 0.76])
st.append(PageBreak())

# ---------- 3 steps ----------
st.append(P("3. Step-by-step documentation", h1))
st.append(P("Each step shows what is done and why, the exact code that was run, its output, and the result. "
            "The code comes straight from <b>src/clustering_and_evaluation.py</b>."))
st.append(P("Part A - Unsupervised learning (clustering)", h2))
for n in range(1, 13):
    why, result = STEP_TEXT[n]
    if n == 6:
        st.append(P("Part B - Model evaluation", h2))
    if n == 10:
        st.append(P("Part C - Why do the models perform differently?", h2))
    st += [CondPageBreak(75 * mm), P(f"Step {n}: {titles[n]}", h2), P(why),
           P("CODE", label), Preformatted(wrap(CODE[n]), code_style),
           P("OUTPUT", label), Preformatted(wrap(LOG[str(n)]), out_style)]
    for cap, rows, widths in EXTRA.get(n, []):
        st += [CondPageBreak(60 * mm), P(cap, label)] + table(rows, widths)   # keep caption and table together
    for name, cap, w in FIGS.get(n, []):
        st += image(name, cap, w)
    st += callout(f"<b>Result:</b> {result}")

# ---------- 4 discussion ----------
gain = lambda m: abl["3 financial features"][m] - abl["All 14 features"][m]
st += [PageBreak(), P("4. Discussion: why the models perform differently", h1),
       P("The test results in Part B rank the models, but a ranking does not explain itself. This section uses the "
         "experiments of Steps 10-12 to explain <i>why</i> each model performs as it does, how the features each "
         "model relies on differ, and how much better any model could realistically be.")]

st += [P("4.1 Why Logistic Regression comes first", h2),
       P(f"Logistic Regression has the highest cross-validated AUC ({f3(cv['Logistic Regression']['roc_auc_mean'])}) "
         f"and the highest test AUC ({f3(lrt['ROC_AUC'])}, with Random Forest at {f3(te['Random Forest']['ROC_AUC'])}) "
         "of the three models. The experiments point to four reasons.")]
st += bullets([
    "<b>Its assumption fits the data.</b> Approval depends mainly on a few financial features that act "
    "additively (credit score, income, loan size). A model that is linear in the log-odds has the right shape "
    "for that. With only two inputs, credit score and loan-to-income, it reaches "
    f"{f3(abl['Credit score + loan-to-income']['Logistic Regression'])} AUC, the highest value in Figure 12, and "
    "adding the other twelve columns does not improve it.",
    "<b>It copes with useless columns.</b> Regularisation (C = 0.1) shrinks the coefficients of irrelevant "
    f"columns almost to zero (Age {bi['Age']['LR_coefficient']:+.3f}, Education "
    f"{bi['Education']['LR_coefficient']:+.3f}, Loan_Term_Months {bi['Loan_Term_Months']['LR_coefficient']:+.3f}), "
    f"so its AUC stays between {min(abl[k]['Logistic Regression'] for k in abl):.3f} and "
    f"{max(abl[k]['Logistic Regression'] for k in abl):.3f} whatever the feature set. It is not perfect: Gender "
    f"({bi['Gender']['LR_coefficient']:+.2f}) and the city dummies (about 0.10 to 0.16 in size) still receive small "
    "coefficients that are only noise.",
    "<b>It needs little data.</b> Its training and validation curves meet "
    f"(gap {lc['Logistic Regression']['train'][-1] - lc['Logistic Regression']['val'][-1]:.2f} at full size), "
    f"and the validation AUC is already {f3(lc['Logistic Regression']['val'][1])} with 131 training rows against "
    f"{f3(lc['Logistic Regression']['val'][-1])} with 329. With 412 training rows it has all the data it can use.",
    "<b>It is already close to the ceiling</b> of what this data allows (Section 4.5).",
])
st.append(P(f"<b>An honest caveat.</b> The advantage is small. Logistic Regression beats Random Forest by "
            f"{pair['Logistic Regression - Random Forest']['Mean_AUC_diff']:+.3f} AUC and KNN by "
            f"{pair['Logistic Regression - KNN']['Mean_AUC_diff']:+.3f} on average, but the corrected p-values are "
            f"{pair['Logistic Regression - Random Forest']['p_value']:.2f} and "
            f"{pair['Logistic Regression - KNN']['p_value']:.2f}, so the ranking is <b>not statistically "
            "established</b> with 515 applicants. The defensible conclusion is that Logistic Regression is at least "
            "as good as the other models here and is simpler and easier to explain, not that it is proven better."))

st += [P("4.2 Why KNN comes last", h2),
       P(f"KNN has the lowest cross-validated AUC ({f3(cv['KNN']['roc_auc_mean'])}) and the lowest test accuracy "
         f"({f3(te['KNN']['Accuracy'])}). Four properties of the method explain it.")]
st += bullets([
    "<b>Distance treats every feature as equally important.</b> The distance is computed over all 14 features, "
    "but only credit score and a few financial features carry signal. The other columns add random distance and "
    "blur the neighbourhoods. Restricting KNN to the three financial features raises its AUC from "
    f"{f3(abl['All 14 features']['KNN'])} to {f3(abl['3 financial features']['KNN'])} "
    f"({gain('KNN'):+.3f}), a repair similar to Random Forest ({gain('Random Forest'):+.3f}) and about twice "
    f"Logistic Regression's ({gain('Logistic Regression'):+.3f}). Credit score alone is too little for it "
    f"({f3(abl['Credit score only']['KNN'])}): it needs income and loan size as well.",
    "<b>It memorises.</b> With distance weighting its training AUC is 1.00 at every training size, because "
    f"each training point is its own nearest neighbour. The gap to the validation AUC is "
    f"{lc['KNN']['train'][-1] - lc['KNN']['val'][-1]:.2f}, the largest of the three models.",
    "<b>It is weak where data are sparse.</b> In the two lowest credit-score quartiles it makes "
    f"{pct(err_credit_rows[0]['KNN'])} and {pct(err_credit_rows[1]['KNN'])} errors, against "
    f"{pct(err_credit_rows[0]['Logistic Regression'])} and {pct(err_credit_rows[1]['Logistic Regression'])} for "
    f"Logistic Regression, and {pct(errw['Unemployed']['KNN'])} against "
    f"{pct(errw['Unemployed']['Logistic Regression'])} for the small group of unemployed applicants. Local methods "
    "have few neighbours to learn from in such regions.",
    "<b>Its tuning may be incomplete.</b> The best setting, k = 31, is at the edge of the searched range "
    "(5, 11, 21, 31). A wider search with even more smoothing might raise KNN a little; the other models' best "
    "settings were inside their ranges.",
])

st += [P("4.3 Why Random Forest sits in the middle", h2),
       P(f"Random Forest has a cross-validated AUC of {f3(cv['Random Forest']['roc_auc_mean'])} with default "
         f"settings and {f3(tn['Random Forest']['CV_AUC'])} after tuning, and the best test recall of the three "
         f"models ({f3(te['Random Forest']['Recall'])}).")]
st += bullets([
    "<b>It is flexible but data-hungry.</b> Its validation AUC rises from "
    f"{f3(lc['Random Forest']['val'][0])} with 65 rows to {f3(lc['Random Forest']['val'][-1])} with 329 and is "
    "still going up, while the gap to its training score narrows from "
    f"{lc['Random Forest']['train'][0] - lc['Random Forest']['val'][0]:.2f} to "
    f"{lc['Random Forest']['train'][-1] - lc['Random Forest']['val'][-1]:.2f}. It is the one model that more "
    "data would probably help most; this was not tested with more data.",
    "<b>Tuning pushed it towards simplicity.</b> The default forest (depth 8) memorises the training data "
    f"(training AUC 1.00 against a validation AUC of {f3(cv['Random Forest']['roc_auc_mean'])}); the best tuned "
    "forest is shallow (depth 4). The validation curve in Figure 7 shows why: depth beyond 4 raises the "
    "training score but not the validation score.",
    "<b>Its main strength has little to work on.</b> Forests can capture interactions and non-linear effects. "
    "Here the data show no sign of them: a linear model matches the forest, and the forest is not better than "
    "Logistic Regression in any credit-score or loan-to-income band of the error analysis.",
    "<b>It also suffers from weak features.</b> With only the three financial features its AUC is "
    f"{f3(abl['3 financial features']['Random Forest'])}, against {f3(abl['All 14 features']['Random Forest'])} "
    "with all 14, and it spreads importance over correlated columns such as Age.",
])

st += [P("4.4 How feature importance varies across the models", h2),
       P("Figure 11 and the tables in Step 10 show what each model relies on. The picture has one clear result and "
         "several differences that are mostly noise or measurement artefacts.")]
st += bullets([
    f"<b>Agreement: credit score.</b> It is first for every model and every method. Shuffling it lowers the "
    f"test AUC by {f3(perm['Credit_Score']['Logistic Regression'])}, {f3(perm['Credit_Score']['KNN'])} and "
    f"{f3(perm['Credit_Score']['Random Forest'])}, whereas no other feature lowers any model's AUC by more than "
    f"{max(perm[k][m] for k in perm if k != 'Credit_Score' for m in M3):.3f}.",
    "<b>Different mechanisms use different secondary features.</b> Logistic Regression combines Annual_Income "
    f"({f3(perm['Annual_Income']['Logistic Regression'])}) and Loan_Amount "
    f"({f3(perm['Loan_Amount']['Logistic Regression'])}) with opposite signs, which amounts to a loan-to-income "
    "ratio. A forest and KNN cannot form a ratio directly, so they pick up the same information through Age "
    f"({f3(perm['Age']['Random Forest'])} and {f3(perm['Age']['KNN'])}), which is correlated with income.",
    "<b>Correlated features share importance.</b> Age, income and loan amount move together (r = 0.74 between "
    "Age and income in Week 1). If a model can substitute one for another, shuffling one column hurts less, so "
    "the importance of each is understated. This is one reason why the rankings agree so little "
    f"(Spearman {f2(agree['Perm. Logistic Regression']['Perm. KNN'])} to "
    f"{f2(agree['Perm. KNN']['Perm. Random Forest'])} between models).",
    "<b>The importance measures themselves differ.</b> Coefficient size depends on the scale of the feature (the "
    "Unemployed dummy has a large coefficient but almost no effect on AUC); impurity importance favours "
    "columns with many distinct values (it ranks Loan_Term_Months and Education 5th and 6th although shuffling "
    f"them changes nothing); and Logistic Regression's own coefficient and permutation rankings agree well "
    f"(Spearman {f2(agree['Perm. Logistic Regression']['LR |coefficient|'])}) while the forest's impurity and "
    f"permutation rankings agree only moderately ({f2(agree['Perm. Random Forest']['RF impurity'])}).",
    "<b>Noise limits what can be said.</b> The typical shuffle-to-shuffle noise is "
    f"{noise['Logistic Regression']:.3f} to {noise['KNN']:.3f} AUC on a 103-row test set. Apart from credit "
    "score, almost every difference between models or features is inside about two times that noise. The only "
    "result that can be trusted is that credit score dominates, followed by the financial pair (income, loan "
    "size).",
])

st += [P("4.5 The ceiling: why no model can do much better here", h2),
       P("Because the dataset was generated in Week 1, the rule that produced the approval label is known: a "
         "logistic function of credit score, income, the loan-to-income ratio and unemployment, from which each "
         "label was <i>drawn at random</i>. Applying that rule to the cleaned data gives a rough upper limit for any "
         "model. It is approximate, because the cleaning in Week 1 capped and filled some values, and it can only "
         "be computed for synthetic data; with real data the ceiling is unknown.")]
st += table([
    ["Measure", "True rule (ceiling)", "Logistic Regression"],
    ["AUC on all 515 applicants", f3(ceil["AUC"]), f"{f3(lr_oof['AUC'])} (out-of-fold)"],
    ["Accuracy on all 515 applicants", f3(ceil["Accuracy"]), f"{f3(lr_oof['Accuracy'])} (out-of-fold)"],
    ["AUC on the 103 test applicants", f3(ceil["AUC_test_rows"]), f3(lrt["ROC_AUC"])],
], [0.40, 0.30, 0.30])
st += bullets([
    "<b>The best model is at the ceiling</b> within the uncertainty of the estimate (about 0.02 AUC on 515 "
    "applicants). The modest scores in Part B reflect the data, not a weak model.",
    f"<b>The mistakes are shared.</b> {share(0):.1f}% of applicants are right for all three models and "
    f"{share(3):.1f}% are wrong for all three; only {share(1) + share(2):.1f}% split the models. Pairwise "
    "disagreement is only 7% to 12%. Different models fail on the same people.",
    f"<b>The hard cases contradict their own profile.</b> The {nw['3']} applicants wrong for all three models "
    f"were approved {hard['Actually approved (%)']['Wrong_for_all_3']:.0f}% of the time, against "
    f"{hard['Actually approved (%)']['Correct_for_all_3']:.0f}% for the group that is always right, although "
    f"credit score ({hard['Credit score']['Wrong_for_all_3']:.0f} vs {hard['Credit score']['Correct_for_all_3']:.0f}) "
    "and loan-to-income are almost the same. This is what random label noise looks like.",
    f"<b>Inside a credit-score band the models add almost nothing.</b> In the lowest three quartiles the "
    f"Logistic Regression error ({pct(err_credit_rows[0]['Logistic Regression'])}, "
    f"{pct(err_credit_rows[1]['Logistic Regression'])}, {pct(err_credit_rows[2]['Logistic Regression'])}) is "
    f"within one percentage point of the best constant guess ({pct(R['base_credit'][1])} and "
    f"{pct(R['base_credit'][2])}) in the second and third quartile, and worse than it in the lowest "
    f"({pct(R['base_credit'][0])}). The skill comes from separating high from low credit scores.",
])

st += [P("4.6 What would actually improve performance", h2)]
st += bullets([
    "<b>More informative features, not more complex models.</b> Three model families with different "
    "mechanisms all stop at the same score and fail on the same applicants.",
    "<b>Domain features for the flexible models.</b> A loan-to-income ratio lifts every model above credit score "
    "alone, and removing uninformative columns helps KNN and Random Forest; both are cheap to do.",
    "<b>More data, if Random Forest is the goal.</b> Its learning curve is the only one still rising.",
    "<b>A decision threshold chosen by cost.</b> At threshold 0.5 the best model wrongly approves "
    f"{pct(cm['fp'] / (cm['fp'] + cm['tn']))} of rejected applicants in the test set (Step 9 shows the trade-off).",
    "<b>Statistical care when comparing models.</b> Use repeated cross-validation and a paired test before "
    "claiming that one model is better; differences of 0.02 to 0.04 AUC are within noise here.",
])

# ---------- 5 summary ----------
st += [CondPageBreak(90 * mm), P("5. Summary of findings", h1)]
st += table([
    ["Question", "Answer"],
    ["How many natural groups are there?",
     f"Weak evidence for {K}: older, higher-income applicants versus younger, lower-income ones "
     f"(silhouette {kt[K]['Silhouette']:.2f}). The data is one continuous cloud, not separate islands."],
    ["Do K-Means and Hierarchical clustering agree?", f"Moderately (ARI {R['ari']:.2f}); they cut the cloud in "
                                                      "different places."],
    ["Are the clusters meaningful?", f"Yes, in a limited way: approval is {c0['Approval_%']:.0f}% vs "
                                     f"{c1['Approval_%']:.0f}%, although approval was not used for clustering."],
    ["Best classifier?", f"{BEST}: cross-validated AUC {f3(tn[BEST]['CV_AUC'])}, test AUC {f3(lrt['ROC_AUC'])}, "
                         f"accuracy {f3(lrt['Accuracy'])}, F1 {f3(lrt['F1'])}."],
    ["Is it good?", "Modestly. It beats the always-approve baseline clearly on AUC (+0.26) and accuracy "
                    "(+0.11) but only slightly on F1 (+0.03). It is close to the best score the data allow."],
    ["Which models overfit?", "KNN (training AUC 1.00 at every size) and the untuned Random Forest "
                              "(1.00 vs 0.71 validation). Logistic Regression does not."],
    ["Why is Logistic Regression best?", "The data are roughly linear in a few features, so a simple model "
                                         "fits; it ignores useless columns and needs little data. The lead is small "
                                         f"(p = {pair['Logistic Regression - Random Forest']['p_value']:.2f} to "
                                         f"{pair['Logistic Regression - KNN']['p_value']:.2f}) and not "
                                         "statistically established."],
    ["Do the models use the same features?", "They agree on credit score (AUC drop 0.18-0.22) and disagree on "
                                             "the rest: Logistic Regression uses income and loan size, KNN and "
                                             "Random Forest use Age as a proxy. Most secondary differences are "
                                             "within noise."],
    ["Could a better model do much better?", f"Probably not: the true rule gives AUC {f3(ceil['AUC'])} against "
                                             f"{f3(lr_oof['AUC'])} for Logistic Regression, and "
                                             f"{share(3):.0f}% of applicants are wrong for all three models."],
    ["How to choose the threshold?", "0.4 for fewer missed good applicants (recall 0.94), 0.6 for fewer "
                                     "wrongly approved ones (precision 0.78): a business decision."],
], [0.32, 0.68])

# ---------- 5 limitations ----------
st += [P("6. Limitations and next steps", h1)]
st += bullets([
    "<b>Weak cluster structure.</b> With a silhouette of about 0.23, the two clusters are a convenient "
    "split of one continuous group, not clearly separate populations. They should be used as descriptive "
    "segments, not as hard categories.",
    "<b>Small, synthetic data.</b> 515 rows and a 103-row test set mean that scores move by several "
    "percentage points with a different split; differences between the tuned models are within noise.",
    "<b>Scaling before splitting.</b> The Week 1 features were scaled on the whole dataset, which leaks a "
    "little information from the test rows. A Scikit-learn Pipeline would remove this.",
    "<b>Possible circularity.</b> Because the data is synthetic and the approval label was generated from "
    "income and credit score, the link between clusters and approval is partly built in. On real data it "
    "would need independent confirmation.",
    "<b>The ceiling is approximate.</b> It applies the known generating rule to cleaned data whose values were "
    "partly capped or filled in Week 1, and its AUC has an uncertainty of about 0.02. It exists only for "
    "synthetic data.",
    "<b>Importance and ablation results are indicative.</b> Permutation importance is computed on only 103 test "
    "rows, the ablation and learning-curve differences are 0.01 to 0.02 AUC, and the out-of-fold predictions "
    "use hyperparameters that were chosen on the training set. None of these differences was tested for "
    "significance, except the model comparison in Step 11c.",
    "<b>Next steps:</b> try DBSCAN and Gaussian mixtures, t-SNE or UMAP for visualisation, use a Pipeline, "
    "add calibration of probabilities, and repeat the evaluation on a real dataset.",
])

# ---------- appendix ----------
st += [P("Appendix: repository and how to run", h1),
       Preformatted(
           "week3-unsupervised-learning-evaluation/\n"
           "  README.md\n"
           "  requirements.txt\n"
           "  data/loan_applicants_clean.csv           cleaned data from Week 1\n"
           "  src/clustering_and_evaluation.py         PCA, clustering, evaluation, tuning, thresholds\n"
           "  src/build_report.py                      builds this PDF\n"
           "  results/                                 tables (CSV), charts (PNG), clustered data, log\n"
           "  models/best_classifier.joblib            saved best classifier\n"
           "  report/                                  this PDF", code_style),
       P("To run: <b>pip install -r requirements.txt</b>, then <b>python3 src/clustering_and_evaluation.py</b> "
         "from the repository root.")]

doc = SimpleDocTemplate(PDF, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                        bottomMargin=18 * mm, title="Week 3 - Unsupervised Learning and Model Evaluation",
                        author=AUTHOR)
doc.build(st, onFirstPage=footer, onLaterPages=footer)
print("Built", PDF)
