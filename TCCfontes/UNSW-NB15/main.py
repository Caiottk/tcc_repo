"""
===========================================================================
UNSW-NB15 — Classificação Binária  (Normal × Ataque)
Modelos : LS-SVM (PSO) × BiLSTM
Split   : Oficial  UNSW_NB15_training-set.csv / UNSW_NB15_testing-set.csv
===========================================================================
Saídas em resultados/:
  • roc_unsw_nb15.{png,pdf}          – Curva ROC comparativa
  • prc_unsw_nb15.{png,pdf}          – Curva Precision-Recall comparativa
  • fpr_unsw_nb15.{png,pdf}          – Gráfico de barras FPR
  • fpr_ttest_boxplot.{png,pdf}      – Boxplot FPR × fold (teste t)
  • lssvm_hparams.json               – C* e σ* encontrados pelo PSO
  • tabelas_latex.tex                – Tabelas prontas para o LaTeX
  • report.log                       – Log completo da execução
===========================================================================
Uso rápido (desenvolvimento):
  python main.py --pso-particles 10 --pso-iters 10 --pso-subsample 0.15
"""

import os
import sys
import json
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib
import scipy.stats as stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)

import pyswarms.single as ps

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

# ── Reprodutibilidade ────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Caminhos ─────────────────────────────────────────────────────────
TRAIN_CSV  = "UNSW_NB15_training-set.csv"
TEST_CSV   = "UNSW_NB15_testing-set.csv"
OUTPUT_DIR = "resultados"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW = 10  # tamanho da janela deslizante para o BiLSTM

# ── Log de execução (_Tee) ───────────────────────────────────────────
_LOG_PATH = os.path.join(OUTPUT_DIR, "report.log")
_log_file = open(_LOG_PATH, "w", encoding="utf-8")


class _Tee:
    """Duplica sys.stdout para terminal e arquivo de log."""
    def __init__(self, *streams): self._streams = streams
    def write(self, data):
        for s in self._streams: s.write(data); s.flush()
    def flush(self):
        for s in self._streams: s.flush()
    def fileno(self): return self._streams[0].fileno()


sys.stdout = _Tee(sys.__stdout__, _log_file)

# ── Argparse ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="UNSW-NB15 — LS-SVM × BiLSTM")
parser.add_argument("--pso-particles", type=int,   default=20,
                    help="Número de partículas do PSO (default: 20)")
parser.add_argument("--pso-iters",     type=int,   default=30,
                    help="Iterações do PSO (default: 30)")
parser.add_argument("--pso-subsample", type=float, default=0.20,
                    help="Fração do treino usada no PSO (default: 0.20)")
args = parser.parse_args()

PSO_PARTICLES = args.pso_particles
PSO_ITERS     = args.pso_iters
PSO_SUBSAMPLE = args.pso_subsample
N_COMPONENTS  = 500
CV_SPLITS     = 3
TTEST_SPLITS  = 5

# =====================================================================
# CLASSES E FUNÇÕES AUXILIARES
# =====================================================================


class LSSVMClassifier(BaseEstimator, ClassifierMixin):
    """LS-SVM escalável via Nyström + RidgeClassifier.

    FORMULAÇÃO (Suykens & Vandewalle, 1999)
    ───────────────────────────────────────
    Resolve o problema primal LS-SVM:
        min  (1/2)||w||² + (C/2) Σᵢ ξᵢ²
        s.t. yᵢ(wᵀφ(xᵢ) + b) = 1 − ξᵢ  ∀i

    Para escala (N >> 10⁴), aproxima-se φ via Nyström:
        (ΦᵀΦ + I/C)w = Φᵀy  ↔  RidgeClassifier(alpha = 1/C)

    Mapeamento de hiperparâmetros:
        σ (RBF bandwidth)  →  γ = 1/(2σ²)
        C (regularização)  →  α_ridge = clip(1/C, ∈[1e-4, 1e4])
        n_components       →  dimensionalidade pós-Nyström

    PROBABILIDADES: Platt scaling via sigmoid(decision_function),
    sem CalibratedClassifierCV, para velocidade sem sacrificar calibração.

    Parâmetros
    ──────────
    C            : regularização (> 0)
    sigma        : largura de banda RBF (> 0)
    n_components : dimensões transformação Nyström (≪ N treino)
    random_state : reprodutibilidade
    """

    def __init__(self, C=1.0, sigma=1.0, n_components=500, random_state=42):
        self.C = C
        self.sigma = sigma
        self.n_components = n_components
        self.random_state = random_state

    def _build_pipeline(self) -> Pipeline:
        gamma = 1.0 / (2.0 * self.sigma ** 2)
        alpha = float(np.clip(1.0 / self.C, 1e-4, 1e4))
        return Pipeline([
            ("nystroem", Nystroem(
                kernel="rbf", gamma=gamma,
                n_components=self.n_components,
                random_state=self.random_state,
            )),
            ("ridge", RidgeClassifier(alpha=alpha, class_weight="balanced")),
        ])

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self._pipe = self._build_pipeline()
        self._pipe.fit(X, y)
        return self

    def predict(self, X):
        return self._pipe.predict(X)

    def decision_function(self, X):
        return self._pipe.decision_function(X)

    def predict_proba(self, X):
        scores = self.decision_function(X).astype(np.float64)
        prob_pos = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1.0 - prob_pos, prob_pos])


def _lssvm_fitness(particles, X_pso, y_pso, n_components, cv_splits=3):
    """Função de fitness PSO: 0.70×FPR_cv + 0.30×(1−F1_cv)."""
    costs = []
    kf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    for row in particles:
        C_p     = float(np.clip(10.0 ** row[0], 1e-2, 1e4))
        sigma_p = float(np.clip(10.0 ** row[1], 1e-3, 1e3))
        fpr_list, f1_list = [], []
        for tr, va in kf.split(X_pso, y_pso):
            clf = LSSVMClassifier(C=C_p, sigma=sigma_p,
                                  n_components=n_components,
                                  random_state=SEED)
            clf.fit(X_pso[tr], y_pso[tr])
            yp = clf.predict(X_pso[va])
            tn, fp, fn, tp = confusion_matrix(y_pso[va], yp).ravel()
            fpr_list.append(fp / (fp + tn + 1e-9))
            f1_list.append(f1_score(y_pso[va], yp, zero_division=0))
        costs.append(0.70 * np.mean(fpr_list) + 0.30 * (1.0 - np.mean(f1_list)))
    return np.array(costs)


def run_pso(X_pso, y_pso, n_components, n_particles, n_iters):
    """Executa PSO para encontrar C* e σ*."""
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
    bounds  = (np.array([-2.0, -3.0]), np.array([4.0, 3.0]))  # log10 C, log10 σ
    optimizer = ps.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=2,
        options=options,
        bounds=bounds,
    )
    best_cost, best_pos = optimizer.optimize(
        lambda p: _lssvm_fitness(p, X_pso, y_pso, n_components),
        iters=n_iters,
        verbose=False,
    )
    C_star     = float(np.clip(10.0 ** best_pos[0], 1e-2, 1e4))
    sigma_star = float(np.clip(10.0 ** best_pos[1], 1e-3, 1e3))
    return C_star, sigma_star, best_cost


def build_bilstm(n_features, window):
    """Cria e compila o modelo BiLSTM."""
    model = Sequential([
        Input(shape=(window, n_features)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def create_windows(X, y, window_size):
    """Cria janelas deslizantes de tamanho `window_size`.
    O rótulo de cada janela é o do último elemento."""
    n_samples = len(X) - window_size + 1
    Xw = np.empty((n_samples, window_size, X.shape[1]), dtype=np.float32)
    yw = np.empty(n_samples, dtype=np.int32)
    for i in range(n_samples):
        Xw[i] = X[i: i + window_size]
        yw[i] = y[i + window_size - 1]
    return Xw, yw


def fpr_from_cm(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn + 1e-9)


def compute_metrics(y_true, y_pred, y_prob, name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "Model": name,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "FPR": fpr_val,
        "AUC-ROC": roc_auc_score(y_true, y_prob),
    }


def save_fig(name):
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close()


# =====================================================================
# 1. CARREGAR E PRÉ-PROCESSAR
# =====================================================================
print("=" * 65)
print("[1/8] Carregando dados...")
print("=" * 65)

df_train = pd.read_csv(TRAIN_CSV)
df_test  = pd.read_csv(TEST_CSV)

print(f"   Treino bruto : {df_train.shape}")
print(f"   Teste  bruto : {df_test.shape}")

# Colunas a descartar (id é índice; attack_cat é multiclasse)
drop_cols = [c for c in ["id", "attack_cat"] if c in df_train.columns]
df_train.drop(columns=drop_cols, inplace=True)
df_test.drop(columns=[c for c in drop_cols if c in df_test.columns], inplace=True)

# Garantir que 'label' existe e é binário (0 = Normal, 1 = Ataque)
assert "label" in df_train.columns, "Coluna 'label' não encontrada!"
y_train = df_train["label"].values.astype(int)
y_test  = df_test["label"].values.astype(int)
df_train.drop(columns="label", inplace=True)
df_test.drop(columns="label", inplace=True)

print(f"   Distribuição treino — Normal: {(y_train==0).sum()}, "
      f"Ataque: {(y_train==1).sum()}")
print(f"   Distribuição teste  — Normal: {(y_test==0).sum()}, "
      f"Ataque: {(y_test==1).sum()}")

# ── Codificar features categóricas ──────────────────────────────────
cat_cols = df_train.select_dtypes(include=["object"]).columns.tolist()
print(f"   Colunas categóricas: {cat_cols}")

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat(
        [df_train[col].astype(str), df_test[col].astype(str)], axis=0
    )
    le.fit(combined)
    df_train[col] = le.transform(df_train[col].astype(str))
    df_test[col]  = le.transform(df_test[col].astype(str))
    encoders[col] = le

# ── Tratar valores ausentes ─────────────────────────────────────────
df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
df_train.fillna(df_train.median(numeric_only=True), inplace=True)
df_test.fillna(df_test.median(numeric_only=True), inplace=True)

# ── Normalizar (StandardScaler) ─────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(df_train.values.astype(np.float32))
X_test  = scaler.transform(df_test.values.astype(np.float32))

n_features = X_train.shape[1]
print(f"   Features finais: {n_features}")
print(f"   X_train: {X_train.shape}  |  X_test: {X_test.shape}")

# =====================================================================
# 2. FORMATANDO ENTRADAS
# =====================================================================
print("\n" + "=" * 65)
print("[2/8] Formatando entradas (estática + sequencial)...")
print("=" * 65)

# ── Entrada estática — LS-SVM ───────────────────────────────────────
X_train_svm = X_train
X_test_svm  = X_test
y_train_svm = y_train.copy()
y_test_svm  = y_test.copy()

# ── Entrada sequencial — BiLSTM (janela deslizante) ─────────────────
X_train_seq, y_train_seq = create_windows(X_train, y_train, WINDOW)
X_test_seq,  y_test_seq  = create_windows(X_test,  y_test,  WINDOW)

print(f"   LS-SVM  — Treino: {X_train_svm.shape}  |  Teste: {X_test_svm.shape}")
print(f"   BiLSTM  — Treino: {X_train_seq.shape}  |  Teste: {X_test_seq.shape}")

# =====================================================================
# 3. PSO — OTIMIZAÇÃO DE HIPERPARÂMETROS DO LS-SVM
# =====================================================================
print("\n" + "=" * 65)
print("[3/8] PSO — Otimizando C e σ do LS-SVM...")
print(f"       Partículas: {PSO_PARTICLES} | Iterações: {PSO_ITERS} | "
      f"Subsample: {PSO_SUBSAMPLE:.0%}")
print("=" * 65)

n_pso   = max(100, int(len(X_train_svm) * PSO_SUBSAMPLE))
idx_pso = np.random.choice(len(X_train_svm), n_pso, replace=False)
X_pso, y_pso = X_train_svm[idx_pso], y_train_svm[idx_pso]

C_star, sigma_star, best_cost = run_pso(
    X_pso, y_pso, N_COMPONENTS, PSO_PARTICLES, PSO_ITERS
)
print(f"   PSO concluído  → C* = {C_star:.4f}  |  σ* = {sigma_star:.4f}  "
      f"|  custo = {best_cost:.6f}")

hparams = {
    "C": C_star,
    "sigma": sigma_star,
    "gamma": 1.0 / (2.0 * sigma_star ** 2),
    "n_components": N_COMPONENTS,
}
with open(os.path.join(OUTPUT_DIR, "lssvm_hparams.json"), "w") as fh:
    json.dump(hparams, fh, indent=2)
print(f"   Salvo: {os.path.join(OUTPUT_DIR, 'lssvm_hparams.json')}")

# =====================================================================
# 4. TREINAR LS-SVM FINAL
# =====================================================================
print("\n" + "=" * 65)
print("[4/8] Treinando LS-SVM (LSSVMClassifier com C* e σ*)...")
print("=" * 65)

lssvm = LSSVMClassifier(C=C_star, sigma=sigma_star,
                        n_components=N_COMPONENTS, random_state=SEED)
lssvm.fit(X_train_svm, y_train_svm)
y_pred_svm = lssvm.predict(X_test_svm)
y_prob_svm = lssvm.predict_proba(X_test_svm)[:, 1]
print("   LS-SVM — treino concluído.")

# =====================================================================
# 5. TREINAR BiLSTM FINAL
# =====================================================================
print("\n" + "=" * 65)
print("[5/8] Treinando BiLSTM...")
print("=" * 65)

# class_weight para compensar desbalanceamento
n_neg_tr = (y_train_seq == 0).sum()
n_pos_tr = (y_train_seq == 1).sum()
class_weight = {0: 1.0, 1: n_neg_tr / max(n_pos_tr, 1)}

bilstm = build_bilstm(n_features, WINDOW)
bilstm.summary()

es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
bilstm.fit(
    X_train_seq, y_train_seq,
    validation_split=0.1,
    epochs=30,
    batch_size=256,
    callbacks=[es],
    class_weight=class_weight,
    verbose=1,
)

y_prob_lstm = bilstm.predict(X_test_seq, batch_size=512).ravel()
y_pred_lstm = (y_prob_lstm >= 0.5).astype(int)
print("   BiLSTM — treino concluído.")

# =====================================================================
# 6. TESTE t-STUDENT PAREADO (TTEST_SPLITS-fold CV)
# =====================================================================
print("\n" + "=" * 65)
print(f"[6/8] Teste t-Student pareado ({TTEST_SPLITS}-fold CV)...")
print("=" * 65)

# CV sobre o conjunto unificado (treino + teste) para máxima representatividade
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

skf = StratifiedKFold(n_splits=TTEST_SPLITS, shuffle=True, random_state=SEED)
fpr_svm_folds, fpr_lstm_folds = [], []

for fold_i, (tr, va) in enumerate(skf.split(X_all, y_all), 1):
    print(f"   Fold {fold_i}/{TTEST_SPLITS}...")

    # LS-SVM fold
    clf_fold = LSSVMClassifier(C=C_star, sigma=sigma_star,
                               n_components=N_COMPONENTS, random_state=SEED)
    clf_fold.fit(X_all[tr], y_all[tr])
    fpr_svm_folds.append(fpr_from_cm(y_all[va], clf_fold.predict(X_all[va])))

    # BiLSTM fold — janelar dados do fold
    Xw_tr, yw_tr = create_windows(X_all[tr], y_all[tr], WINDOW)
    Xw_va, yw_va = create_windows(X_all[va], y_all[va], WINDOW)
    n_neg_f = (yw_tr == 0).sum()
    n_pos_f = (yw_tr == 1).sum()
    cw_fold = {0: 1.0, 1: n_neg_f / max(n_pos_f, 1)}
    lstm_fold = build_bilstm(X_all.shape[1], WINDOW)
    es_fold   = EarlyStopping(monitor="val_loss", patience=3,
                              restore_best_weights=True)
    lstm_fold.fit(Xw_tr, yw_tr, validation_split=0.1,
                  epochs=20, batch_size=256,
                  callbacks=[es_fold], class_weight=cw_fold, verbose=0)
    yp_fold = (lstm_fold.predict(Xw_va, batch_size=512).ravel() >= 0.5).astype(int)
    fpr_lstm_folds.append(fpr_from_cm(yw_va, yp_fold))

fpr_svm_folds  = np.array(fpr_svm_folds)
fpr_lstm_folds = np.array(fpr_lstm_folds)
t_stat, p_val  = stats.ttest_rel(fpr_svm_folds, fpr_lstm_folds)

print(f"\n   FPR LS-SVM (folds): {np.round(fpr_svm_folds, 4)}")
print(f"   FPR BiLSTM (folds): {np.round(fpr_lstm_folds, 4)}")
print(f"   t = {t_stat:.4f}  |  p = {p_val:.6f}")
if p_val < 0.05:
    print("   → Diferença ESTATISTICAMENTE SIGNIFICATIVA (α=0.05)")
else:
    print("   → Diferença NÃO significativa (α=0.05)")

# =====================================================================
# 7. MÉTRICAS FINAIS
# =====================================================================
print("\n" + "=" * 65)
print("[7/8] Calculando métricas finais...")
print("=" * 65)

m_svm  = compute_metrics(y_test_svm, y_pred_svm, y_prob_svm, "LS-SVM")
m_lstm = compute_metrics(y_test_seq, y_pred_lstm, y_prob_lstm, "BiLSTM")

results = pd.DataFrame([m_svm, m_lstm])
print(results.to_string(index=False))

# =====================================================================
# 8. GRÁFICOS + TABELA LATEX
# =====================================================================
print("\n" + "=" * 65)
print("[8/8] Gerando figuras e tabela LaTeX...")
print("=" * 65)

# ── 8a. Curva ROC ───────────────────────────────────────────────────
fpr_svm_c, tpr_svm_c, _   = roc_curve(y_test_svm, y_prob_svm)
fpr_lstm_c, tpr_lstm_c, _ = roc_curve(y_test_seq, y_prob_lstm)

plt.figure(figsize=(7, 5))
plt.plot(fpr_svm_c,  tpr_svm_c,
         label=f"LS-SVM  (AUC = {m_svm['AUC-ROC']:.4f})", linewidth=1.5)
plt.plot(fpr_lstm_c, tpr_lstm_c,
         label=f"BiLSTM  (AUC = {m_lstm['AUC-ROC']:.4f})", linewidth=1.5)
plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Aleatório")
plt.xlabel("Taxa de Falso Positivo (FPR)")
plt.ylabel("Taxa de Verdadeiro Positivo (TPR)")
plt.title("Curva ROC — UNSW-NB15")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
save_fig("roc_unsw_nb15")
print("   ✓ roc_unsw_nb15.png/.pdf")

# ── 8b. Curva Precision-Recall ───────────────────────────────────────
pre_svm_c, rec_svm_c, _   = precision_recall_curve(y_test_svm, y_prob_svm)
pre_lstm_c, rec_lstm_c, _ = precision_recall_curve(y_test_seq, y_prob_lstm)

plt.figure(figsize=(7, 5))
plt.plot(rec_svm_c, pre_svm_c, label="LS-SVM", linewidth=1.5)
plt.plot(rec_lstm_c, pre_lstm_c, label="BiLSTM", linewidth=1.5)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall — UNSW-NB15")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()
save_fig("prc_unsw_nb15")
print("   ✓ prc_unsw_nb15.png/.pdf")

# ── 8c. Gráfico de barras — FPR ─────────────────────────────────────
plt.figure(figsize=(5, 4))
models_list = ["LS-SVM", "BiLSTM"]
fprs   = [m_svm["FPR"], m_lstm["FPR"]]
colors = ["#1f77b4", "#ff7f0e"]
bars   = plt.bar(models_list, fprs, color=colors, width=0.45, edgecolor="black")
for bar, val in zip(bars, fprs):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{val:.4f}",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
    )
plt.ylabel("False Positive Rate (FPR)")
plt.title("Comparação de FPR — UNSW-NB15")
plt.ylim(0, max(fprs) * 1.30 + 0.01)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
save_fig("fpr_unsw_nb15")
print("   ✓ fpr_unsw_nb15.png/.pdf")

# ── 8d. Boxplot t-test ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
bp = ax.boxplot(
    [fpr_svm_folds, fpr_lstm_folds],
    labels=["LS-SVM", "BiLSTM"],
    patch_artist=True,
    medianprops=dict(color="navy", linewidth=2),
)
for patch, color in zip(bp["boxes"], ["#cce5ff", "#ffd6a5"]):
    patch.set_facecolor(color)
ax.set_ylabel("FPR")
ax.set_title(f"FPR por Fold — t={t_stat:.3f}, p={p_val:.4f}")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save_fig("fpr_ttest_boxplot")
print("   ✓ fpr_ttest_boxplot.png/.pdf")

# ── 8e. Tabelas LaTeX ────────────────────────────────────────────────
latex_metrics = r"""\begin{table}[htbp]
\centering
\caption{Resultados da Classificação Binária — UNSW-NB15}
\label{tab:unsw_nb15_results}
\begin{tabular}{lccccc}
\toprule
\textbf{Modelo} & \textbf{Precisão} & \textbf{Recall} & \textbf{F1-Score} & \textbf{FPR} & \textbf{AUC-ROC} \\
\midrule
"""
for m in [m_svm, m_lstm]:
    latex_metrics += (
        f"  {m['Model']} & "
        f"{m['Precision']:.4f} & "
        f"{m['Recall']:.4f} & "
        f"{m['F1-Score']:.4f} & "
        f"{m['FPR']:.4f} & "
        f"{m['AUC-ROC']:.4f} \\\\\n"
    )
latex_metrics += r"""\bottomrule
\end{tabular}
\end{table}
"""

latex_ttest = r"""\begin{table}[htbp]
\centering
\caption{Teste $t$ Pareado — FPR por Fold (UNSW-NB15)}
\label{tab:ttest_unsw}
\begin{tabular}{lccccccc}
\toprule
\textbf{Modelo} & \textbf{Fold 1} & \textbf{Fold 2} & \textbf{Fold 3} & \textbf{Fold 4} & \textbf{Fold 5} & \textbf{Média} & \textbf{p-valor} \\
\midrule
"""
for name, folds in [("LS-SVM", fpr_svm_folds), ("BiLSTM", fpr_lstm_folds)]:
    row = f"  {name}"
    for v in folds:
        row += f" & {v:.4f}"
    row += f" & {folds.mean():.4f}"
    row += f" & {p_val:.4f}" if name == "BiLSTM" else " & —"
    row += " \\\\\n"
    latex_ttest += row
latex_ttest += r"""\bottomrule
\end{tabular}
\end{table}
"""

tex_path = os.path.join(OUTPUT_DIR, "tabelas_latex.tex")
with open(tex_path, "w", encoding="utf-8") as fh:
    fh.write(latex_metrics + "\n" + latex_ttest)
print(f"   ✓ tabelas_latex.tex")

# ── Resumo final ─────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"  Figuras     : {os.path.abspath(OUTPUT_DIR)}")
print(f"  JSON        : {os.path.join(OUTPUT_DIR, 'lssvm_hparams.json')}")
print(f"  LaTeX       : {tex_path}")
print(f"  Log         : {os.path.abspath(_LOG_PATH)}")
print("=" * 65)
print("\n✅  Script concluído com sucesso!")

sys.stdout = sys.__stdout__
_log_file.close()
print(f"\n📄  Log salvo em: {os.path.abspath(_LOG_PATH)}")
