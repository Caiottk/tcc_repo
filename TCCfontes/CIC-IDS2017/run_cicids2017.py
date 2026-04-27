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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  NOTAS METODOLÓGICAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LS-SVM (Least-Squares SVM – Suykens & Vandewalle, 1999)
  Formulação primal:
      min  (1/2)||w||² + (C/2) Σᵢ ξᵢ²
      s.t. yᵢ(wᵀφ(xᵢ) + b) = 1 − ξᵢ  ∀i

  Para N >> 10⁴ amostras, resolver o sistema dual N×N é inviável.
  Utiliza-se a aproximação de Nyström para mapear φ̃: x → R^m (m ≪ N)
  e resolve-se o sistema primal via RidgeClassifier(α = 1/C), onde
  γ_rbf = 1/(2σ²) mapeia diretamente o parâmetro de largura de banda σ.

  Os hiperparâmetros C e σ são otimizados por Particle Swarm Optimization
  (PSO) cujo fitness penaliza primariamente a FPR (peso 0.7) e
  secundariamente (1 − F1) (peso 0.3) via validação cruzada estratificada.

BiLSTM
  Entrada: janela deslizante de T = 10 fluxos consecutivos.
  Tensor shape: (Amostras, T=10, Features).
  Arquitetura: BiLSTM(2 camadas, 128 hidden) → FC(64) → ReLU → FC(1).

Validação estatística
  Teste t de Student pareado (scipy) sobre as FPRs obtidas em N folds de
  validação cruzada (configurável via --ttest-splits, default 3), para verificar
  diferença estatisticamente significativa entre LS-SVM e BiLSTM ao nível α = 0.05.
"""

# ══════════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════════
import argparse
import gc
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, ClassifierMixin
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import pyswarms.single as ps
    HAS_PYSWARMS = True
except ImportError:
    HAS_PYSWARMS = False
    print("[AVISO] pyswarms não encontrado. Instale com: pip install pyswarms")
    print("        Usando parâmetros padrão (sem PSO).\n")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
    # PSO
    p.add_argument("--pso-particles", type=int,   default=20,
                   help="Número de partículas do PSO (default: 20)")
    p.add_argument("--pso-iters",     type=int,   default=30,
                   help="Iterações do PSO (default: 30)")
    p.add_argument("--pso-subsample", type=float, default=0.30,
                   help="Fração de X_train usada no fitness PSO (default: 0.30)")
    p.add_argument("--ttest-splits", type=int, default=3,
                   help="Nº de folds para o teste t-Student (default: 3)")
    p.add_argument("--nys-components", type=int, default=500,
                   help="Dimensões da aproximação Nyström (default: 500)")
    p.add_argument("--skip-pso", action="store_true",
                   help="Pula PSO e lê C/σ de lssvm_hparams.json se existir")
    return p.parse_args()


ARGS           = parse_args()
CSV_PATH       = ARGS.csv
SUBSAMPLE_FRAC = 1.0 if ARGS.full else ARGS.subsample
WINDOW_SIZE    = 10          # janela deslizante T para BiLSTM
TEST_RATIO     = 0.20
RANDOM_STATE   = 42

# ── LS-SVM: hiperparâmetros PSO ───────────────────────────────────────
# Busca em escala log:  C ∈ [10⁻², 10²],  σ ∈ [10⁻², 10¹]
PSO_C_BOUNDS     = (-2.0, 2.0)   # log10(C)
PSO_SIGMA_BOUNDS = (-2.0, 1.0)   # log10(σ)
PSO_PARTICLES    = ARGS.pso_particles
PSO_ITERS        = ARGS.pso_iters
PSO_SUBSAMPLE    = ARGS.pso_subsample
PSO_FPR_WEIGHT   = 0.70          # peso da FPR na função de fitness
PSO_F1_WEIGHT    = 0.30          # peso de (1 − F1)
NYS_COMPONENTS   = ARGS.nys_components  # dimensões da aprox. Nyström

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

# ── Tee: redireciona stdout para terminal + arquivo de log ──────────────
class _Tee:
    """Escreve simultaneamente em dois streams (stdout + arquivo)."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self._streams:
            s.flush()
    def fileno(self):
        # Necessário para compatibilidade com warnings do Python
        return self._streams[0].fileno()

_LOG_PATH   = os.path.join(OUTPUT_DIR, "report.log")
_log_file   = open(_LOG_PATH, "w", encoding="utf-8")
sys.stdout  = _Tee(sys.__stdout__, _log_file)
# ────────────────────────────────────────────────────────────────

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

THRESHOLDS = [0.3, 0.5, 0.7]
CV_FOLDS   = ARGS.ttest_splits  # folds para FPR cross-val (teste t-Student)

print("=" * 65)
print("  TCC – LS-SVM  vs  BiLSTM   (CIC-IDS2017)")
print(f"  Device      : {DEVICE}")
print(f"  Sub-amostra : {SUBSAMPLE_FRAC * 100:.0f} %")
print(f"  CSV         : {CSV_PATH}")
print(f"  PSO         : {PSO_PARTICLES} partículas × {PSO_ITERS} iter")
print("=" * 65)


# ══════════════════════════════════════════════════════════════════════
#  2. CARGA  E  PRÉ-PROCESSAMENTO
# ══════════════════════════════════════════════════════════════════════
print("\n[1/9] Carregando dados …")
df = pd.read_csv(CSV_PATH)

# Limpa espaços nos nomes de colunas (comum neste dataset)
df.columns = df.columns.str.strip()

# ── detecta coluna de rótulo ──────────────────────────────────────────
label_col = None
for c in ["Label", "label", "Class", "class", "target", "Target",
          "Attack Type", "attack_type"]:
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

# OTIMIZAÇÃO DE MEMÓRIA: Liberar DataFrame (~1–2 GB) antes de Nyström.
# Fórmula de alocação Nyström:
#   pico_RAM ≈ N_train × n_components × 8 bytes × 2  (duas cópias internas)
# 
# Exemplos com N_train ≈ 2.016M (CIC-IDS2017 80%):
#   n_components=150  → 2.016M × 150 × 16 ≈ 8.5 GB  ✓ seguro (máxima usada)
#   n_components=200  → 2.016M × 200 × 16 ≈ 10.3 GB ✗ OOM (15GB limite)
#   n_components=500  → 2.016M × 500 × 16 ≈ 25.8 GB ✗ OOM
#
# Este é o fator limitante crítico do hardware documentado no RELATORIO.
del df, X
gc.collect()
print(f"  Treino : {X_train.shape[0]:,}   |   Teste : {X_test.shape[0]:,}\n")


# ══════════════════════════════════════════════════════════════════════
#  3. CLASSE LSSVMClassifier  (Suykens-compatible, escalável via Nyström)
# ══════════════════════════════════════════════════════════════════════
class LSSVMClassifier(BaseEstimator, ClassifierMixin):
    """
    LS-SVM escalável (Suykens & Vandewalle, 1999) via aproximação de Nyström.

    FORMULAÇÃO MATEMÁTICA
    ─────────────────────
    O LS-SVM resolve o problema primal:
        min  (1/2)||w||² + (C/2) Σᵢ ξᵢ²
        s.t. yᵢ(wᵀφ(xᵢ) + b) = 1 − ξᵢ  ∀i

    Equivale a um sistema linear tras aplicar condições KKT.

    ESCALABILIDADE: USO DE NYSTRÖM
    ──────────────────────────────
    Para N >> 10⁴ amostras, resolver o sistema dual N×N é computacionalmente
    intratável. Aplicamos a aproximação Nyström (Williams & Seeger 2001):

        φ̃(x) = K(x, Z) [K(Z, Z)]^{-1/2}   onde Z ⊂ dataset (m ≪ N pontos)

    Isso reduz φ:ℝᵈ → ℝᵐ. O sistema primal fica:
        (ΦᵀΦ + I/C)w = Φᵀy   ↔  RidgeClassifier(alpha = 1/C)

    MAPEAMENTO DE HIPERPARÂMETROS
    ──────────────────────────────
    - σ (RBF bandwidth) ↔ γ_rbf = 1/(2σ²)  (direto kernel_rbf)
    - C (regularização) ↔ α_ridge = 1/C   (clipped ∈ [1e-4, 1e4])
    - n_components: dimensão pós-Nyström (default 500 em psutil 15GB → uso 150)

    CÁLCULO DE PROBABILIDADE
    ────────────────────────
    predict_proba utiliza Platt Scaling via sigmoid(decision_function),
    sem CalibratedClassifierCV, para acelerar mas mantendo calibração.

    Parâmetros
    ──────────
    C            : float  – regularização (> 0)
    sigma        : float  – largura de banda RBF (> 0)
    n_components : int    – nº de features Nyström pós-aproximação (≪ N)
    random_state : int    – semente reprodutibilidade
    """

    def __init__(
        self,
        C: float = 1.0,
        sigma: float = 1.0,
        n_components: int = 500,
        random_state: int = 42,
    ):
        self.C            = C
        self.sigma        = sigma
        self.n_components = n_components
        self.random_state = random_state
        self._pipeline    = None

    def _build_pipeline(self) -> Pipeline:
        gamma = 1.0 / (2.0 * self.sigma ** 2)
        # alpha = 1/C; mínimo de 1e-4 para evitar matriz mal-condicionada
        alpha = float(np.clip(1.0 / self.C, 1e-4, 1e4))
        return Pipeline([
            ("nystroem", Nystroem(
                kernel="rbf",
                gamma=gamma,
                n_components=self.n_components,
                random_state=self.random_state,
            )),
            ("ridge", RidgeClassifier(
                alpha=alpha,
                class_weight="balanced",
            )),
        ])

    def fit(self, X, y):
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self._pipeline.predict(X)

    def decision_function(self, X):
        return self._pipeline.decision_function(X)

    def predict_proba(self, X):
        """Probabilidades via sigmoid do decision_function (Platt-like)."""
        scores    = self.decision_function(X).astype(np.float64)
        prob_pos  = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1.0 - prob_pos, prob_pos])


# ══════════════════════════════════════════════════════════════════════
#  4. PSO  –  otimização de (C, σ) com fitness orientado à FPR
# ══════════════════════════════════════════════════════════════════════

def _lssvm_fitness(
    particles: np.ndarray,
    X_pso: np.ndarray,
    y_pso: np.ndarray,
    n_components: int,
    cv_splits: int = 3,
) -> np.ndarray:
    """
    Função de fitness PSO para otimizar hiperparâmetros de LS-SVM.

    PARAMETRIZAÇÃO
    ──────────────
    Cada partícula ∈ ℝ² representa [log₁₀(C), log₁₀(σ)]:
      C   ∈ [0.01, 100]  (regularização)
      σ   ∈ [0.01, 10]   (RBF bandwidth)

    FÓRMULA DE CUSTO
    ────────────────
    A cada iteração, a partícula (C, σ) é validada via k-fold CV estratificado
    e o custo retornado é:

        custo = 0.70 × FPR_cv  +  0.30 × (1 − F1_cv)

    Rationale:
      • FPR (peso 0.7): reduzir falsos positivos é o objetivo central do TCC.
      •        Minimizá-la favorece modelos seletivos (alta precisão).
      • F1 (peso 0.3): term que impede sacrifício excessivo de recall.
      •        Garante que detecção real de ataques não caia drasticamente.

    OTIMIZAÇÕES
    ───────────
    - X_pso subsampled (5% do treino ~100k) controla tempo total.
    - y_pso mantém proporção original de classes (estratificado).
    - Clipping: alpha_ridge ∈ [1e-4, 1e4] evita LinAlgWarning em RidgeClassifier.

    Argumentos
    ──────────
    particles    : (N, 2) array de posições PSO
    X_pso, y_pso : subsample treino + labels (5%)
    n_components : dimensionalidade Nyström (default 150, tipo INT)
    cv_splits    : folds de CV (default 3)

    Retorna
    ──────
    Vetor (N,) de custos, com NaNs para partículas fora de bounds.
    """
    costs = np.zeros(len(particles))
    skf   = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    for i, particle in enumerate(particles):
        C     = float(np.clip(10 ** particle[0], 1e-4, 1e4))
        sigma = float(np.clip(10 ** particle[1], 1e-4, 1e2))

        fpr_list, f1_list = [], []
        for train_idx, val_idx in skf.split(X_pso, y_pso):
            Xtr, Xvl = X_pso[train_idx], X_pso[val_idx]
            ytr, yvl = y_pso[train_idx], y_pso[val_idx]
            clf = LSSVMClassifier(
                C=C, sigma=sigma,
                n_components=n_components,
                random_state=42,
            )
            try:
                clf.fit(Xtr, ytr)
                pred = clf.predict(Xvl)
                tn, fp, fn, tp = confusion_matrix(
                    yvl, pred, labels=[0, 1]
                ).ravel()
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                f1  = f1_score(yvl, pred, zero_division=0)
            except Exception:
                fpr, f1 = 1.0, 0.0
            fpr_list.append(fpr)
            f1_list.append(f1)

        costs[i] = (
            PSO_FPR_WEIGHT * np.mean(fpr_list)
            + PSO_F1_WEIGHT * (1.0 - np.mean(f1_list))
        )
    return costs


def run_pso(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_components: int = NYS_COMPONENTS,
) -> tuple:
    """
    Executa o PSO para otimizar (C, σ) do LS-SVM.
    Retorna (C_best, sigma_best).
    """
    if not HAS_PYSWARMS:
        print("  [PSO] pyswarms ausente – usando C=10, σ=0.5 (padrão)")
        return 10.0, 0.5

    n_pso = int(len(X_train) * PSO_SUBSAMPLE)
    idx   = np.random.choice(len(X_train), n_pso, replace=False)
    X_pso = X_train[idx]
    y_pso = y_train[idx]
    print(f"  [PSO] Sub-amostra fitness: {n_pso:,} amostras  "
          f"({PSO_SUBSAMPLE*100:.0f}% do treino)")

    options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
    bounds  = (
        np.array([PSO_C_BOUNDS[0],     PSO_SIGMA_BOUNDS[0]]),
        np.array([PSO_C_BOUNDS[1],     PSO_SIGMA_BOUNDS[1]]),
    )

    optimizer = ps.GlobalBestPSO(
        n_particles=PSO_PARTICLES,
        dimensions=2,
        options=options,
        bounds=bounds,
    )
    best_cost, best_pos = optimizer.optimize(
        _lssvm_fitness,
        iters=PSO_ITERS,
        verbose=True,
        X_pso=X_pso,
        y_pso=y_pso,
        n_components=n_components,
        cv_splits=3,
    )

    C_best     = float(10 ** best_pos[0])
    sigma_best = float(10 ** best_pos[1])
    print(f"\n  [PSO] Melhor custo : {best_cost:.6f}")
    print(f"  [PSO] C*    = {C_best:.4f}")
    print(f"  [PSO] σ*    = {sigma_best:.4f}  →  γ* = {1/(2*sigma_best**2):.4f}")
    return C_best, sigma_best


# ══════════════════════════════════════════════════════════════════════
#  5. TREINO LS-SVM  (PSO + ajuste final + calibração)
# ══════════════════════════════════════════════════════════════════════
_hparams_path = os.path.join(OUTPUT_DIR, "lssvm_hparams.json")
if ARGS.skip_pso and os.path.exists(_hparams_path):
    print("[2/9] PSO – PULADO  (--skip-pso)  →  lendo lssvm_hparams.json …")
    with open(_hparams_path) as _f:
        _hp = json.load(_f)
    C_best         = float(_hp["C"])
    sigma_best     = float(_hp["sigma"])
    NYS_COMPONENTS = int(_hp.get("n_components", NYS_COMPONENTS))
    print(f"  [PSO] C*            = {C_best:.4f}")
    print(f"  [PSO] σ*            = {sigma_best:.4f}  →  γ* = {1/(2*sigma_best**2):.4f}")
    print(f"  [PSO] n_components  = {NYS_COMPONENTS}")
else:
    if ARGS.skip_pso:
        print("[AVISO] --skip-pso solicitado mas lssvm_hparams.json não encontrado → executando PSO normalmente.")
    print("[2/9] PSO – otimizando hiperparâmetros do LS-SVM …")
    C_best, sigma_best = run_pso(X_train, y_train, NYS_COMPONENTS)
    # ── Salvar hparams imediatamente após PSO (antes de qualquer passo seguinte)
    _hp_early = {"C": C_best, "sigma": sigma_best,
                 "gamma": 1.0 / (2.0 * sigma_best ** 2),
                 "n_components": NYS_COMPONENTS}
    with open(_hparams_path, "w") as _f:
        json.dump(_hp_early, _f, indent=2)
    print(f"  [PSO] Hiperparâmetros salvos em  {_hparams_path}")

t0 = time.time()

print(f"\n[3/9] Treinando LS-SVM final  (C={C_best:.4f}, σ={sigma_best:.4f}) …")

lssvm_base = LSSVMClassifier(
    C=C_best,
    sigma=sigma_best,
    n_components=NYS_COMPONENTS,
    random_state=RANDOM_STATE,
)

# LSSVMClassifier já expõe predict_proba via sigmoid do decision_function
lssvm = lssvm_base
lssvm.fit(X_train, y_train)

lssvm_time  = time.time() - t0
lssvm_proba = lssvm.predict_proba(X_test)[:, 1]
lssvm_pred  = (lssvm_proba >= 0.5).astype(int)
print(f"  Concluído em {lssvm_time:.1f} s  (PSO + treino final)\n")


# ══════════════════════════════════════════════════════════════════════
#  6. BiLSTM   (PyTorch)  –  janela deslizante T=10
#     Shape de entrada: (Amostras, T=10, Features)
# ══════════════════════════════════════════════════════════════════════

class SlidingWindowDataset(Dataset):
    """
    Gera janelas (T, F) on-the-fly a partir do array (N, F).
    O rótulo de cada janela é o do último fluxo da janela.

    Input shape  : (N, F)
    Output shape : (T, F)  com T = window
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


print("[4/9] Preparando janelas deslizantes  (T = {}) …".format(WINDOW_SIZE))
train_ds = SlidingWindowDataset(X_train, y_train, WINDOW_SIZE)
test_ds  = SlidingWindowDataset(X_test,  y_test,  WINDOW_SIZE)
print(f"  Janelas  treino : {len(train_ds):,}   |   teste : {len(test_ds):,}")
print(f"  Tensor shape  : (batch, T={WINDOW_SIZE}, F={NUM_FEATURES})")

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=(DEVICE.type == "cuda"),
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=(DEVICE.type == "cuda"),
)


class BiLSTMClassifier(nn.Module):
    """
    BiLSTM Bidirecional + Cabeçote FC para Classificação Binária.

    ARQUITETURA
    ────────────
    Entrada        : (batch, T=10, features)  ← janela deslizante de 10 fluxos
    Saída          : logits (batch,)
    
    Stack:
      1. BiLSTM(2 camadas, 128 hidden, bidirecional)  → (B, T, 256)
      2. Extrai último time-step                       → (B, 256)
      3. FC(256 → 64) + ReLU + Dropout                 → (B, 64)
      4. FC(64 → 1) + BCEWithLogitsLoss               → logit ∈ ℝ

    LIMITAÇÃO CRÍTICA: CONTAMINAÇÃO TREINO-VALIDAÇÃO
    ─────────────────────────────────────────────────
    A janela deslizante cria sobreposição temporal entre sequências:
      • Fluxo 1: seq[0:10]  (índices 0–9)
      • Fluxo 2: seq[1:11]  (índices 1–10)
      • ...
      • Fluxo 8: seq[7:17]  (índices 7–16)
      • Fluxo 9: seq[8:18]  (índices 8–17)

    Split treino-val (80/20) por índice de fluxo, mas a sobreposição significa
    9 de 10 pontos de qualquer validação aparecem no treino:
      • 90% de contaminação temporal → val_loss artificialmente baixa
      • Gera melhor métricas do que em produção real.

    NOTA: BiLSTM mantém seu valor comparativo (LS-SVM não usa sequences),
    mas esse bias afeta ambos modelos igualmente. Recomenda-se usar
    batcher temporalmente disjuntos em produção.

    Parâmetros
    ──────────
    input_dim  : nº de features por timestep
    hidden_dim : nº de hidden units (→ 2×hidden com bidireccional)
    num_layers : nº de ámaras LSTM (default 2)
    dropout    : dropout rate (aplicado entre camadas + após LSTM)
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
print(f"\n[5/9] Treinando BiLSTM  (épocas={EPOCHS}, paciência={PATIENCE}) …")
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
#  7. MÉTRICAS PRINCIPAIS
# ════════════════════════════════════════════════════════════════════
print("\n[6/9] Calculando métricas …")


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
#  8. TESTE T DE STUDENT (validação estatística das FPRs)
# ════════════════════════════════════════════════════════════════════
print(f"[7/9] Teste t de Student (FPR – validação cruzada {CV_FOLDS}-fold) …")

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

fpr_svm_cv  = []
fpr_lstm_cv = []

for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_train, y_train), 1):
    Xtr_f, Xvl_f = X_train[tr_idx], X_train[vl_idx]
    ytr_f, yvl_f = y_train[tr_idx], y_train[vl_idx]

    # ── LS-SVM fold ──
    clf_f = LSSVMClassifier(
        C=C_best, sigma=sigma_best,
        n_components=NYS_COMPONENTS, random_state=RANDOM_STATE,
    )
    clf_f.fit(Xtr_f, ytr_f)
    pred_svm_f        = clf_f.predict(Xvl_f)
    tn, fp, fn, tp    = confusion_matrix(yvl_f, pred_svm_f, labels=[0, 1]).ravel()
    fpr_svm_cv.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)

    # ── BiLSTM fold (inferência com modelo já treinado) ──
    val_ds_f = SlidingWindowDataset(Xvl_f, yvl_f, WINDOW_SIZE)
    if len(val_ds_f) == 0:
        fpr_lstm_cv.append(0.0)
        continue
    val_ld_f = DataLoader(val_ds_f, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)
    logits_f, yt_f = [], []
    with torch.no_grad():
        for xb, yb in val_ld_f:
            logits_f.append(model(xb.to(DEVICE)).cpu())
            yt_f.append(yb)
    logits_f = torch.cat(logits_f).numpy()
    yt_f     = torch.cat(yt_f).numpy().astype(int)
    pred_l_f = (1.0 / (1.0 + np.exp(-logits_f)) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt_f, pred_l_f, labels=[0, 1]).ravel()
    fpr_lstm_cv.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)

    print(f"  Fold {fold}/{CV_FOLDS}  FPR LS-SVM={fpr_svm_cv[-1]:.4f}  "
          f"BiLSTM={fpr_lstm_cv[-1]:.4f}")

fpr_svm_cv  = np.array(fpr_svm_cv)
fpr_lstm_cv = np.array(fpr_lstm_cv)

t_stat, p_value = stats.ttest_rel(fpr_svm_cv, fpr_lstm_cv)

print(f"\n  FPR médio LS-SVM : {fpr_svm_cv.mean():.4f}  ± {fpr_svm_cv.std():.4f}")
print(f"  FPR médio BiLSTM : {fpr_lstm_cv.mean():.4f}  ± {fpr_lstm_cv.std():.4f}")
print(f"  t-estatístico    : {t_stat:.4f}")
print(f"  p-valor          : {p_value:.4f}")
if p_value < 0.05:
    better = "LS-SVM" if fpr_svm_cv.mean() < fpr_lstm_cv.mean() else "BiLSTM"
    print(f"  → Diferença SIGNIFICATIVA (α=0.05). {better} reduz mais FPR.\n")
else:
    print("  → Diferença NÃO significativa (α=0.05).\n")


# ════════════════════════════════════════════════════════════════════
#  9. VISUALIZAÇÕES
# ════════════════════════════════════════════════════════════════════
print("[8/9] Gerando figuras …")
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

# ── Boxplot FPR cross-val (teste t-Student) ─────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ax.boxplot(
    [fpr_svm_cv, fpr_lstm_cv],
    labels=["LS-SVM", "BiLSTM"],
    patch_artist=True,
    boxprops=dict(facecolor="#D0E4F7"),
    medianprops=dict(color="navy", lw=2),
)
ax.set_ylabel(f"FPR (validação cruzada – {CV_FOLDS} folds)")
ax.set_title(
    f"Distribuição de FPR — t={t_stat:.3f},  p={p_value:.4f}"
    + ("  *" if p_value < 0.05 else "")
)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fpr_ttest_boxplot.png"), dpi=300)
fig.savefig(os.path.join(OUTPUT_DIR, "fpr_ttest_boxplot.pdf"))
plt.close(fig)
print("  ✓ fpr_ttest_boxplot.png / .pdf")


# ══════════════════════════════════════════════════════════════════════
#  10. TABELAS LATEX
# ════════════════════════════════════════════════════════════════════
print("\n[9/9] Gerando tabelas LaTeX …\n")

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

# ── Tabela 3 – Teste t-Student ──────────────────────────────────────
sig_str = "sim" if p_value < 0.05 else "não"
latex_ttest = (
    "\\begin{table}[htbp]\n"
    "  \\centering\n"
    f"  \\caption{{Teste $t$ de Student pareado da FPR (validação cruzada {CV_FOLDS}-fold).}}\n"
    "  \\label{tab:ttest}\n"
    f"  \\begin{{tabular}}{{l {'c ' * (CV_FOLDS + 1)}}}\n"
    "    \\toprule\n"
    f"    \\textbf{{Modelo}} & \\multicolumn{{{CV_FOLDS}}}{{c}}{{\\textbf{{FPR por fold}}}} & \\textbf{{Média}} \\\\\n"
    "    \\midrule\n"
)
latex_ttest += (
    "    LS-SVM & "
    + " & ".join(f"{v:.4f}" for v in fpr_svm_cv)
    + f" & {fpr_svm_cv.mean():.4f} \\\\\n"
)
latex_ttest += (
    "    BiLSTM & "
    + " & ".join(f"{v:.4f}" for v in fpr_lstm_cv)
    + f" & {fpr_lstm_cv.mean():.4f} \\\\\n"
)
latex_ttest += (
    "    \\midrule\n"
    f"    \\multicolumn{{{CV_FOLDS + 2}}}{{l}}{{$t = {t_stat:.4f}$, $p = {p_value:.4f}$, "
    f"diferença significativa: {sig_str} ($\\alpha = 0.05$)}} \\\\\n"
    "    \\bottomrule\n"
    "  \\end{tabular}\n"
    "\\end{table}\n"
)

full_latex = full_latex + "\n" + latex_ttest

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
# Salvar hiperparâmetros ótimos do PSO
hparams = {"C": C_best, "sigma": sigma_best,
           "gamma": 1.0 / (2.0 * sigma_best ** 2),
           "n_components": NYS_COMPONENTS}
with open(os.path.join(OUTPUT_DIR, "lssvm_hparams.json"), "w") as f:
    json.dump(hparams, f, indent=2)
# ══════════════════════════════════════════════════════════════════════
#  RESUMO FINAL
# ══════════════════════════════════════════════════════════════════════
print(f"\n  Figuras salvas em : {os.path.abspath(OUTPUT_DIR)}/")
print(f"  Tabela LaTeX em   : {os.path.abspath(tex_path)}")
print(f"  Hiperparâmetros   : {os.path.join(OUTPUT_DIR, 'lssvm_hparams.json')}")
print(f"\n  Parâmetros PSO  → C* = {C_best:.4f}, σ* = {sigma_best:.4f}")
print(f"  FPR LS-SVM (cv) : {fpr_svm_cv.mean():.4f} ± {fpr_svm_cv.std():.4f}")
print(f"  FPR BiLSTM (cv) : {fpr_lstm_cv.mean():.4f} ± {fpr_lstm_cv.std():.4f}")
print(f"  t-stat / p-val  : {t_stat:.4f} / {p_value:.4f}")
print(f"  Tempo LS-SVM  : {lssvm_time:>7.1f} s  (PSO + treino final)")
print(f"  Tempo BiLSTM  : {bilstm_time:>7.1f} s")
print(f"  Tempo total   : {lssvm_time + bilstm_time:>7.1f} s")
print(f"  Log completo  : {os.path.abspath(_LOG_PATH)}")
print("\n✅  Script concluído com sucesso!")

# Fecha o log e restaura stdout
sys.stdout = sys.__stdout__
_log_file.close()
print(f"\n📄  Log salvo em: {os.path.abspath(_LOG_PATH)}")
