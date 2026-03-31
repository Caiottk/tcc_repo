"""
===========================================================================
UNSW-NB15 — Classificação Binária  (Normal × Ataque)
Modelos : LS-SVM  ×  BiLSTM
Split   : Oficial  UNSW_NB15_training-set.csv / UNSW_NB15_testing-set.csv
===========================================================================
Saídas:
  • figures/roc_unsw_nb15.png    – Curva ROC comparativa
  • figures/prc_unsw_nb15.png    – Curva Precision-Recall comparativa
  • figures/fpr_unsw_nb15.png    – Gráfico de barras FPR
  • Tabela LaTeX no console
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)

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

# ── Caminhos e hiperparâmetros ───────────────────────────────────────
TRAIN_CSV = "UNSW_NB15_training-set.csv"
TEST_CSV = "UNSW_NB15_testing-set.csv"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

WINDOW = 10  # tamanho da janela deslizante para o BiLSTM

# =====================================================================
# 1. CARREGAR E PRÉ-PROCESSAR
# =====================================================================
print("=" * 65)
print("[1/7] Carregando dados...")
print("=" * 65)

df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)

print(f"   Treino bruto : {df_train.shape}")
print(f"   Teste  bruto : {df_test.shape}")

# Colunas a descartar (id é índice; attack_cat é multiclasse)
drop_cols = [c for c in ["id", "attack_cat"] if c in df_train.columns]
df_train.drop(columns=drop_cols, inplace=True)
df_test.drop(columns=[c for c in drop_cols if c in df_test.columns], inplace=True)

# Garantir que 'label' existe e é binário (0 = Normal, 1 = Ataque)
assert "label" in df_train.columns, "Coluna 'label' não encontrada!"
y_train = df_train["label"].values.astype(int)
y_test = df_test["label"].values.astype(int)
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
    df_test[col] = le.transform(df_test[col].astype(str))
    encoders[col] = le

# ── Tratar valores ausentes ─────────────────────────────────────────
df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
df_train.fillna(df_train.median(numeric_only=True), inplace=True)
df_test.fillna(df_test.median(numeric_only=True), inplace=True)

# ── Normalizar (StandardScaler) ─────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(df_train.values.astype(np.float32))
X_test = scaler.transform(df_test.values.astype(np.float32))

n_features = X_train.shape[1]
print(f"   Features finais: {n_features}")
print(f"   X_train: {X_train.shape}  |  X_test: {X_test.shape}")

# =====================================================================
# 2. FORMATAR ENTRADAS
# =====================================================================
print("\n" + "=" * 65)
print("[2/7] Formatando entradas (estática + sequencial)...")
print("=" * 65)

# ── Entrada estática — LS-SVM ───────────────────────────────────────
X_train_svm = X_train
X_test_svm = X_test
y_train_svm = y_train.copy()
y_test_svm = y_test.copy()


# ── Entrada sequencial — BiLSTM (janela deslizante) ─────────────────
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


X_train_seq, y_train_seq = create_windows(X_train, y_train, WINDOW)
X_test_seq, y_test_seq = create_windows(X_test, y_test, WINDOW)

print(f"   LS-SVM   — Treino: {X_train_svm.shape}  |  Teste: {X_test_svm.shape}")
print(f"   BiLSTM   — Treino: {X_train_seq.shape}  |  Teste: {X_test_seq.shape}")

# =====================================================================
# 3. TREINAR LS-SVM
# =====================================================================
print("\n" + "=" * 65)
print("[3/7] Treinando LS-SVM (LinearSVC + squared hinge)...")
print("=" * 65)

lsvm_base = LinearSVC(
    loss="squared_hinge",  # aproximação LS-SVM (mínimos quadrados)
    dual=True,
    max_iter=10000,
    random_state=SEED,
)
lsvm = CalibratedClassifierCV(estimator=lsvm_base, cv=3)
lsvm.fit(X_train_svm, y_train_svm)

y_pred_svm = lsvm.predict(X_test_svm)
y_prob_svm = lsvm.predict_proba(X_test_svm)[:, 1]
print("   LS-SVM — treino concluído.")

# =====================================================================
# 4. TREINAR BiLSTM
# =====================================================================
print("\n" + "=" * 65)
print("[4/7] Treinando BiLSTM...")
print("=" * 65)

model = Sequential(
    [
        Input(shape=(WINDOW, n_features)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)
model.summary()

es = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

history = model.fit(
    X_train_seq,
    y_train_seq,
    validation_split=0.1,
    epochs=30,
    batch_size=256,
    callbacks=[es],
    verbose=1,
)

y_prob_lstm = model.predict(X_test_seq, batch_size=512).ravel()
y_pred_lstm = (y_prob_lstm >= 0.5).astype(int)
print("   BiLSTM — treino concluído.")

# =====================================================================
# 5. MÉTRICAS
# =====================================================================
print("\n" + "=" * 65)
print("[5/7] Calculando métricas...")
print("=" * 65)


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


m_svm = compute_metrics(y_test_svm, y_pred_svm, y_prob_svm, "LS-SVM")
m_lstm = compute_metrics(y_test_seq, y_pred_lstm, y_prob_lstm, "BiLSTM")

results = pd.DataFrame([m_svm, m_lstm])
print(results.to_string(index=False))

# =====================================================================
# 6. GRÁFICOS
# =====================================================================
print("\n" + "=" * 65)
print("[6/7] Gerando figuras...")
print("=" * 65)

# ── 6a. Curva ROC comparativa ───────────────────────────────────────
fpr_svm_c, tpr_svm_c, _ = roc_curve(y_test_svm, y_prob_svm)
fpr_lstm_c, tpr_lstm_c, _ = roc_curve(y_test_seq, y_prob_lstm)

plt.figure(figsize=(7, 5))
plt.plot(
    fpr_svm_c, tpr_svm_c,
    label=f"LS-SVM  (AUC = {m_svm['AUC-ROC']:.4f})", linewidth=1.5,
)
plt.plot(
    fpr_lstm_c, tpr_lstm_c,
    label=f"BiLSTM  (AUC = {m_lstm['AUC-ROC']:.4f})", linewidth=1.5,
)
plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Aleatório")
plt.xlabel("Taxa de Falso Positivo (FPR)")
plt.ylabel("Taxa de Verdadeiro Positivo (TPR)")
plt.title("Curva ROC — UNSW-NB15")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "roc_unsw_nb15.png"), dpi=300)
plt.close()
print("   ✓ roc_unsw_nb15.png")

# ── 6b. Curva Precision-Recall (PRC) ────────────────────────────────
pre_svm_c, rec_svm_c, _ = precision_recall_curve(y_test_svm, y_prob_svm)
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
plt.savefig(os.path.join(FIG_DIR, "prc_unsw_nb15.png"), dpi=300)
plt.close()
print("   ✓ prc_unsw_nb15.png")

# ── 6c. Gráfico de barras — FPR ─────────────────────────────────────
plt.figure(figsize=(5, 4))
models_list = ["LS-SVM", "BiLSTM"]
fprs = [m_svm["FPR"], m_lstm["FPR"]]
colors = ["#1f77b4", "#ff7f0e"]
bars = plt.bar(models_list, fprs, color=colors, width=0.45, edgecolor="black")
for bar, val in zip(bars, fprs):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{val:.4f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
plt.ylabel("False Positive Rate (FPR)")
plt.title("Comparação de FPR — UNSW-NB15")
plt.ylim(0, max(fprs) * 1.30 + 0.01)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fpr_unsw_nb15.png"), dpi=300)
plt.close()
print("   ✓ fpr_unsw_nb15.png")

# =====================================================================
# 7. TABELA LATEX
# =====================================================================
print("\n" + "=" * 65)
print("[7/7] Tabela LaTeX — Resultados UNSW-NB15")
print("=" * 65)

latex = r"""
\begin{table}[htbp]
\centering
\caption{Resultados da Classificação Binária — UNSW-NB15}
\label{tab:unsw_nb15_results}
\begin{tabular}{lccccc}
\toprule
\textbf{Modelo} & \textbf{Precisão} & \textbf{Recall} & \textbf{F1-Score} & \textbf{FPR} & \textbf{AUC-ROC} \\
\midrule
"""

for m in [m_svm, m_lstm]:
    latex += (
        f"  {m['Model']} & "
        f"{m['Precision']:.4f} & "
        f"{m['Recall']:.4f} & "
        f"{m['F1-Score']:.4f} & "
        f"{m['FPR']:.4f} & "
        f"{m['AUC-ROC']:.4f} \\\\\n"
    )

latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

print(latex)
print("=" * 65)
print("Concluído! Figuras salvas em:", os.path.abspath(FIG_DIR))
print("=" * 65)
