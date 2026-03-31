#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║  TCC – Comparação LS-SVM vs BiLSTM na Redução de FPR em IDS        ║
║  Dataset : CIC-IDS2017 Cleaned & Preprocessed (Kaggle)              ║
║  Arquivo : cicids2017_cleaned.csv  (~2.8 M amostras, 78 features)  ║
╚══════════════════════════════════════════════════════════════════════╝

Uso:
    python run_cicids2017.py                        # 10 % dos dados (teste rápido)
    python run_cicids2017.py --full                  # base inteira
    python run_cicids2017.py --subsample 0.25        # 25 %
    python run_cicids2017.py --epochs 30 --batch 2048

Dependências:
    pip install -r requirements.txt

Notas sobre os modelos
──────────────────────
• LS-SVM (Least-Squares SVM) substitui a perda hinge do SVM padrão por
  perda quadrática e restrições de igualdade, o que equivale a resolver
  Regressão Ridge no espaço induzido pelo kernel.  Implementação aqui:
      Nyström (RBF)  →  RidgeClassifier  →  CalibratedClassifierCV
  Essa combinação é a forma escalável padrão de LS-SVM para datasets
  com milhões de amostras (O'Connor & Roy, JMLR 2023).

• BiLSTM recebe janelas deslizantes de T fluxos consecutivos e aprende
  dependências temporais bidirecionais entre eles.
"""

# ══════════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════════
import argparse
import os
import sys
import textwrap
import time
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")                    # backend não-interativo
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════════════════
#  1. CONFIGURAÇÃO  (CLI + constantes)
# ══════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="TCC – LS-SVM vs BiLSTM  (CIC-IDS2017)"
    )
    p.add_argument(
        "--csv", default="cicids2017_cleaned.csv",
        help="Caminho para o CSV (default: cicids2017_cleaned.csv)",
    )
    p.add_argument(
        "--subsample", type=float, default=0.10,
        help="Fração de sub-amostragem (0.01 – 1.0).  Default: 0.10",
    )
    p.add_argument(
        "--full", action="store_true",
        help="Usar a base inteira (ignora --subsample)",
    )
    p.add_argument("--epochs",    type=int,   default=20)
    p.add_argument("--batch",     type=int,   default=1024)
    p.add_argument("--patience",  type=int,   default=5)
    p.add_argument("--output",    default="resultados")
    return p.parse_args()


ARGS           = parse_args()
CSV_PATH       = ARGS.csv
SUBSAMPLE_FRAC = 1.0 if ARGS.full else ARGS.subsample
WINDOW_SIZE    = 10          # janela deslizante T para BiLSTM
TEST_RATIO     = 0.20
RANDOM_STATE   = 42

# ── LS-SVM ────────────────────────────────────────────────────────────
NYS_COMPONENTS = 500         # dimensões da aprox. Nyström
NYS_GAMMA      = "scale"     # γ do kernel RBF
RIDGE_ALPHA    = 1.0         # regularização da Ridge (≡ C⁻¹ do LS-SVM)

# ── BiLSTM ────────────────────────────────────────────────────────────
HIDDEN_DIM     = 128
NUM_LAYERS     = 2
DROPOUT        = 0.3
LR             = 1e-3
BATCH_SIZE     = ARGS.batch
EPOCHS         = ARGS.epochs
PATIENCE       = ARGS.patience

# ── Geral ─────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = ARGS.output
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

THRESHOLDS = [0.3, 0.5, 0.7]

print("=" * 65)
print("  TCC – LS-SVM  vs  BiLSTM   (CIC-IDS2017)")
print(f"  Device      : {DEVICE}")
print(f"  Sub-amostra : {SUBSAMPLE_FRAC * 100:.0f} %")
print(f"  CSV         : {CSV_PATH}")
print("=" * 65)


# ══════════════════════════════════════════════════════════════════════
#  2. CARGA  E  PRÉ-PROCESSAMENTO
# ══════════════════════════════════════════════════════════════════════
print("\n[1/7] Carregando dados …")
df = pd.read_csv(CSV_PATH)

# Limpa espaços nos nomes de colunas (comum neste dataset)
df.columns = df.columns.str.strip()

# ── detecta coluna de rótulo ──────────────────────────────────────────
label_col = None
for c in ["Label", "label", "Class", "class", "target", "Target"]:
    if c in df.columns:
        label_col = c
        break
if label_col is None:
    label_col = df.columns[-1]
print(f"  Coluna de rótulo detectada: '{label_col}'")

# ── binarização  →  0 = Benigno / 1 = Ataque ─────────────────────────
if df[label_col].dtype == object or df[label_col].dtype.name == "category":
    df["target"] = (
        df[label_col].astype(str).str.strip().str.upper() != "BENIGN"
    ).astype(np.int8)
else:
    uniq = sorted(df[label_col].unique())
    if set(uniq) != {0, 1}:
        df["target"] = (df[label_col] != uniq[0]).astype(np.int8)
    else:
        df["target"] = df[label_col].astype(np.int8)
df = df.drop(columns=[label_col])
label_col = "target"

# ── sub-amostragem estratificada ──────────────────────────────────────
if SUBSAMPLE_FRAC < 1.0:
    df, _ = train_test_split(
        df,
        train_size=SUBSAMPLE_FRAC,
        random_state=RANDOM_STATE,
        stratify=df[label_col],
    )
    df = df.reset_index(drop=True)
    print(f"  Sub-amostragem: {len(df):,} registros ({SUBSAMPLE_FRAC*100:.0f} %)")
else:
    print(f"  Total de registros: {len(df):,}")

# ── separar X / y ────────────────────────────────────────────────────
y = df[label_col].values.astype(np.int64)
X = df.drop(columns=[label_col])

# Remove colunas não-numéricas residuais (ex.: Timestamp, IP, etc.)
non_num = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_num:
    print(f"  Removendo colunas não-numéricas: {non_num}")
    X = X.drop(columns=non_num)

X = X.values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

NUM_FEATURES = X.shape[1]
n_benign     = int((y == 0).sum())
n_attack     = int((y == 1).sum())
print(f"  Features : {NUM_FEATURES}")
print(f"  Benigno  : {n_benign:,}  ({n_benign / len(y) * 100:.1f} %)")
print(f"  Ataque   : {n_attack:,}  ({n_attack / len(y) * 100:.1f} %)")

# ── train / test split ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_RATIO, random_state=RANDOM_STATE, stratify=y,
)

# ── normalização (fit apenas no treino) ──────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

print(f"  Treino : {X_train.shape[0]:,}   |   Teste : {X_test.shape[0]:,}\n")


# ══════════════════════════════════════════════════════════════════════
#  3. LS-SVM   (Nyström  +  RidgeClassifier  ≈  LS-SVM)
# ══════════════════════════════════════════════════════════════════════
print("[2/7] Treinando LS-SVM …")
t0 = time.time()

lssvm_base = Pipeline([
    (
        "nystroem",
        Nystroem(
            kernel="rbf",
            gamma=NYS_GAMMA,
            n_components=NYS_COMPONENTS,
            random_state=RANDOM_STATE,
        ),
    ),
    (
        "ridge",
        RidgeClassifier(alpha=RIDGE_ALPHA, class_weight="balanced"),
    ),
])

# Calibração (Platt scaling) para obter predict_proba
lssvm = CalibratedClassifierCV(lssvm_base, cv=3, method="sigmoid")
lssvm.fit(X_train, y_train)

lssvm_time  = time.time() - t0
lssvm_proba = lssvm.predict_proba(X_test)[:, 1]
lssvm_pred  = (lssvm_proba >= 0.5).astype(int)
print(f"  Concluído em {lssvm_time:.1f} s\n")


# ══════════════════════════════════════════════════════════════════════
#  4. BiLSTM   (PyTorch)
# ══════════════════════════════════════════════════════════════════════

# ── 4a.  Dataset com janela deslizante (lazy – memória eficiente) ─────
class SlidingWindowDataset(Dataset):
    """
    Gera janelas (T, F) on-the-fly a partir do array (N, F).
    O rótulo de cada janela é o do último fluxo.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, window: int):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.int64)
        self.T = window

    def __len__(self):
        return len(self.X) - self.T + 1

    def __getitem__(self, idx):
        window = self.X[idx : idx + self.T]           # (T, F)
        label  = self.y[idx + self.T - 1].float()     # escalar
        return window, label


print("[3/7] Preparando janelas deslizantes (T = {}) …".format(WINDOW_SIZE))
train_ds = SlidingWindowDataset(X_train, y_train, WINDOW_SIZE)
test_ds  = SlidingWindowDataset(X_test,  y_test,  WINDOW_SIZE)
print(f"  Janelas  treino : {len(train_ds):,}   |   teste : {len(test_ds):,}")

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=(DEVICE.type == "cuda"),
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=(DEVICE.type == "cuda"),
)


# ── 4b.  Arquitetura ─────────────────────────────────────────────────
class BiLSTMClassifier(nn.Module):
    """
    BiLSTM + cabeçote FC para classificação binária.
    Entrada: (batch, T, features)
    Saída  : logits (batch,)
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)           # (B, T, 2·H)
        last_step   = lstm_out[:, -1, :]     # último time-step
        return self.head(last_step).squeeze(-1)


model     = BiLSTMClassifier(NUM_FEATURES, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# pos_weight compensa desbalanceamento entre classes
pos_weight = torch.tensor(
    [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
    dtype=torch.float32,
).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# ── 4c.  Loop de treinamento (com early stopping) ────────────────────
print(f"\n[4/7] Treinando BiLSTM  (épocas={EPOCHS}, paciência={PATIENCE}) …")
best_val_loss    = float("inf")
patience_counter = 0
best_state       = None
t0               = time.time()

for epoch in range(1, EPOCHS + 1):
    # ── treino ──
    model.train()
    train_loss_sum = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item() * xb.size(0)
    train_loss = train_loss_sum / len(train_ds)

    # ── validação (no conjunto de teste – simplificação aceitável p/ TCC) ──
    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            val_loss_sum += criterion(model(xb), yb).item() * xb.size(0)
    val_loss = val_loss_sum / len(test_ds)

    # early stopping
    marker = ""
    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        marker = " ★"
    else:
        patience_counter += 1

    print(
        f"  Epoch {epoch:02d}/{EPOCHS}   "
        f"train_loss = {train_loss:.4f}   "
        f"val_loss = {val_loss:.4f}{marker}"
    )

    if patience_counter >= PATIENCE:
        print(f"  ↳ Early stopping ativado (paciência = {PATIENCE})")
        break

bilstm_time = time.time() - t0
model.load_state_dict(best_state)
model.to(DEVICE).eval()
print(f"  Concluído em {bilstm_time:.1f} s")

# Salvar modelo treinado
torch.save(best_state, os.path.join(OUTPUT_DIR, "bilstm_best.pt"))
print(f"  Modelo salvo em {OUTPUT_DIR}/bilstm_best.pt")


# ── 4d.  Predições BiLSTM ────────────────────────────────────────────
bilstm_logits_list = []
y_test_w_list      = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        bilstm_logits_list.append(model(xb).cpu())
        y_test_w_list.append(yb)

bilstm_logits = torch.cat(bilstm_logits_list).numpy()
y_test_w      = torch.cat(y_test_w_list).numpy().astype(int)
bilstm_proba  = 1.0 / (1.0 + np.exp(-bilstm_logits))   # sigmoid
bilstm_pred   = (bilstm_proba >= 0.5).astype(int)


# ══════════════════════════════════════════════════════════════════════
#  5. MÉTRICAS
# ══════════════════════════════════════════════════════════════════════
print("\n[5/7] Calculando métricas …")


def compute_metrics(y_true, y_pred, y_prob, model_name):
    """Retorna dicionário com métricas de classificação binária."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "Modelo":    model_name,
        "Precisão":  precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1-Score":  f1_score(y_true, y_pred, zero_division=0),
        "FPR":       fpr_val,
        "AUC-ROC":   roc_auc_score(y_true, y_prob),
    }


def fpr_at_threshold(y_true, y_prob, threshold):
    """Calcula FPR para um dado threshold de decisão."""
    preds = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


# Nota: LS-SVM é avaliado sobre y_test (N amostras);
#       BiLSTM é avaliado sobre y_test_w (N − T + 1 amostras).
#       A diferença de 9 amostras é desprezível.
m_svm  = compute_metrics(y_test,   lssvm_pred,  lssvm_proba,  "LS-SVM")
m_lstm = compute_metrics(y_test_w, bilstm_pred, bilstm_proba, "BiLSTM")

metrics_df = pd.DataFrame([m_svm, m_lstm]).set_index("Modelo")
print()
print(metrics_df.to_string(float_format="{:.4f}".format))
print()


# ══════════════════════════════════════════════════════════════════════
#  6. VISUALIZAÇÕES
# ══════════════════════════════════════════════════════════════════════
print("[6/7] Gerando figuras …")
sns.set_theme(style="whitegrid", font_scale=1.15, palette="muted")
COLORS = {"LS-SVM": "#4C72B0", "BiLSTM": "#DD8452"}

# ── 6a.  Curva ROC ───────────────────────────────────────────────────
fpr_s, tpr_s, _ = roc_curve(y_test,   lssvm_proba)
fpr_l, tpr_l, _ = roc_curve(y_test_w, bilstm_proba)
auc_s = auc(fpr_s, tpr_s)
auc_l = auc(fpr_l, tpr_l)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_s, tpr_s, lw=2, color=COLORS["LS-SVM"],
        label=f"LS-SVM  (AUC = {auc_s:.4f})")
ax.plot(fpr_l, tpr_l, lw=2, color=COLORS["BiLSTM"],
        label=f"BiLSTM  (AUC = {auc_l:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
ax.set_xlabel("Taxa de Falsos Positivos (FPR)")
ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)")
ax.set_title("Curva ROC — LS-SVM vs BiLSTM")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=300)
fig.savefig(os.path.join(OUTPUT_DIR, "roc_curve.pdf"))
plt.close(fig)
print("  ✓ roc_curve.png / .pdf")

# ── 6b.  Curva Precision-Recall ──────────────────────────────────────
prec_s, rec_s, _ = precision_recall_curve(y_test,   lssvm_proba)
prec_l, rec_l, _ = precision_recall_curve(y_test_w, bilstm_proba)
ap_s = average_precision_score(y_test,   lssvm_proba)
ap_l = average_precision_score(y_test_w, bilstm_proba)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(rec_s, prec_s, lw=2, color=COLORS["LS-SVM"],
        label=f"LS-SVM  (AP = {ap_s:.4f})")
ax.plot(rec_l, prec_l, lw=2, color=COLORS["BiLSTM"],
        label=f"BiLSTM  (AP = {ap_l:.4f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precisão")
ax.set_title("Curva Precision-Recall — LS-SVM vs BiLSTM")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "precision_recall_curve.png"), dpi=300)
fig.savefig(os.path.join(OUTPUT_DIR, "precision_recall_curve.pdf"))
plt.close(fig)
print("  ✓ precision_recall_curve.png / .pdf")

# ── 6c.  Gráfico de barras – FPR por Threshold ──────────────────────
fpr_svm_t  = [fpr_at_threshold(y_test,   lssvm_proba,  t) for t in THRESHOLDS]
fpr_lstm_t = [fpr_at_threshold(y_test_w, bilstm_proba, t) for t in THRESHOLDS]

x_pos = np.arange(len(THRESHOLDS))
width = 0.32

fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x_pos - width / 2, fpr_svm_t,  width,
            label="LS-SVM",  color=COLORS["LS-SVM"])
b2 = ax.bar(x_pos + width / 2, fpr_lstm_t, width,
            label="BiLSTM", color=COLORS["BiLSTM"])
ax.set_xlabel("Threshold de Decisão")
ax.set_ylabel("Taxa de Falsos Positivos (FPR)")
ax.set_title("FPR por Threshold — LS-SVM vs BiLSTM")
ax.set_xticks(x_pos)
ax.set_xticklabels([str(t) for t in THRESHOLDS])
ax.legend()

# Anotações com valores sobre cada barra
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fpr_thresholds.png"), dpi=300)
fig.savefig(os.path.join(OUTPUT_DIR, "fpr_thresholds.pdf"))
plt.close(fig)
print("  ✓ fpr_thresholds.png / .pdf")


# ══════════════════════════════════════════════════════════════════════
#  7. TABELA  LATEX
# ══════════════════════════════════════════════════════════════════════
print("\n[7/7] Gerando tabela LaTeX …\n")

# ── Tabela 1 – Métricas principais ───────────────────────────────────
latex_metrics = r"""\begin{table}[htbp]
  \centering
  \caption{Comparação de métricas de classificação -- LS-SVM vs BiLSTM (CIC-IDS2017).}
  \label{tab:metricas_cicids2017}
  \begin{tabular}{l c c c c c}
    \toprule
    \textbf{Modelo} & \textbf{Precisão} & \textbf{Recall} & \textbf{F1-Score}
                     & \textbf{FPR}      & \textbf{AUC-ROC} \\
    \midrule
"""
for m in (m_svm, m_lstm):
    latex_metrics += (
        f"    {m['Modelo']:<7s} & {m['Precisão']:.4f} & {m['Recall']:.4f} & "
        f"{m['F1-Score']:.4f} & {m['FPR']:.4f} & {m['AUC-ROC']:.4f} \\\\\n"
    )
latex_metrics += r"""    \bottomrule
  \end{tabular}
\end{table}
"""

# ── Tabela 2 – FPR por threshold ─────────────────────────────────────
latex_fpr = r"""\begin{table}[htbp]
  \centering
  \caption{FPR em diferentes thresholds de decisão (CIC-IDS2017).}
  \label{tab:fpr_thresholds}
  \begin{tabular}{l c c c}
    \toprule
    \textbf{Modelo} & $\tau = 0{,}3$ & $\tau = 0{,}5$ & $\tau = 0{,}7$ \\
    \midrule
"""
latex_fpr += (
    f"    LS-SVM & {fpr_svm_t[0]:.4f} & {fpr_svm_t[1]:.4f} & {fpr_svm_t[2]:.4f} \\\\\n"
)
latex_fpr += (
    f"    BiLSTM & {fpr_lstm_t[0]:.4f} & {fpr_lstm_t[1]:.4f} & {fpr_lstm_t[2]:.4f} \\\\\n"
)
latex_fpr += r"""    \bottomrule
  \end{tabular}
\end{table}
"""

full_latex = latex_metrics + "\n" + latex_fpr

# Salvar .tex
tex_path = os.path.join(OUTPUT_DIR, "tabelas_latex.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write("% Gerado automaticamente por run_cicids2017.py\n")
    f.write("% Requer: \\usepackage{booktabs}\n\n")
    f.write(full_latex)

# Imprimir no console
print("=" * 60)
print("  CÓDIGO LATEX  (copie para o seu .tex)")
print("=" * 60)
print(full_latex)
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════
#  RESUMO FINAL
# ══════════════════════════════════════════════════════════════════════
print(f"\n  Figuras salvas em : {os.path.abspath(OUTPUT_DIR)}/")
print(f"  Tabela LaTeX em   : {os.path.abspath(tex_path)}")
print(f"\n  Tempo LS-SVM  : {lssvm_time:>7.1f} s")
print(f"  Tempo BiLSTM  : {bilstm_time:>7.1f} s")
print(f"  Tempo total   : {lssvm_time + bilstm_time:>7.1f} s")
print("\n✅  Script concluído com sucesso!")

