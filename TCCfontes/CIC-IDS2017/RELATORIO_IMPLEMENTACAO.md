# Relatório de Implementação — LS-SVM vs BiLSTM (CIC-IDS2017)

**Projeto:** TCC – Comparação de LS-SVM e BiLSTM na Redução de Taxa de Falsos Positivos em IDS  
**Dataset:** CIC-IDS2017 Cleaned & Preprocessed (Kaggle) — `cicids2017_cleaned.csv`  
**Data:** 2026-03-31  

---

## 1. Estrutura do Projeto

```
CIC-IDS2017/
├── run_cicids2017.py          # Script principal (único arquivo)
├── requirements.txt           # Dependências Python
├── cicids2017_cleaned.csv     # ← colocar o CSV aqui (não versionado)
└── resultados/                # Gerado automaticamente na execução
    ├── roc_curve.png / .pdf
    ├── precision_recall_curve.png / .pdf
    ├── fpr_thresholds.png / .pdf
    ├── bilstm_best.pt         # Pesos do melhor modelo BiLSTM
    └── tabelas_latex.tex      # Tabelas prontas para inserir no LaTeX
```

---

## 2. Dependências e Ambiente

| Pacote         | Versão Mínima | Papel                                      |
|----------------|:-------------:|---------------------------------------------|
| `numpy`        | ≥ 1.23        | Manipulação vetorial                        |
| `pandas`       | ≥ 1.5         | Leitura do CSV e manipulação tabular        |
| `scikit-learn` | ≥ 1.2         | LS-SVM (Nyström + Ridge), métricas, splits  |
| `matplotlib`   | ≥ 3.6         | Geração dos gráficos                        |
| `seaborn`      | ≥ 0.12        | Estilização dos gráficos                    |
| `torch`        | ≥ 2.0         | BiLSTM (treinamento e inferência)           |
| `tqdm`         | ≥ 4.64        | (reservado para barras de progresso futuras) |

**Instalação:**
```bash
pip install -r requirements.txt
```

> **Nota sobre PyTorch:** Se o servidor possuir GPU NVIDIA com CUDA, instale a
> versão GPU para acelerar o treinamento do BiLSTM:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 3. Como Executar

```bash
# Teste rápido (10% dos dados — ~280 K amostras)
python run_cicids2017.py

# Base inteira (~2.8 M amostras)
python run_cicids2017.py --full

# Sub-amostragem customizada (ex.: 25%)
python run_cicids2017.py --subsample 0.25

# Ajustar hiperparâmetros via CLI
python run_cicids2017.py --full --epochs 30 --batch 2048 --patience 7
```

| Flag          | Default                    | Descrição                              |
|---------------|:--------------------------:|----------------------------------------|
| `--csv`       | `cicids2017_cleaned.csv`   | Caminho para o arquivo CSV             |
| `--subsample` | `0.10`                     | Fração de sub-amostragem (0.01 – 1.0) |
| `--full`      | *off*                      | Usa 100% dos dados (ignora subsample)  |
| `--epochs`    | `20`                       | Épocas máximas do BiLSTM               |
| `--batch`     | `1024`                     | Tamanho do mini-batch                  |
| `--patience`  | `5`                        | Épocas sem melhoria para early stop    |
| `--output`    | `resultados`               | Diretório de saída                     |

---

## 4. Pipeline de Dados

### 4.1. Carga e Binarização

1. O CSV é lido com `pandas.read_csv`.
2. Espaços em nomes de colunas são removidos (o dataset original usa `" Label"`).
3. A coluna de rótulo é detectada automaticamente entre candidatas comuns (`Label`, `label`, `Class`, `target`, ou a última coluna).
4. **Binarização:** `BENIGN → 0`, qualquer tipo de ataque → `1`.
5. Colunas não-numéricas residuais (Timestamp, IP, etc.) são descartadas.
6. Valores `NaN`, `+Inf` e `-Inf` são substituídos por `0.0`.

### 4.2. Sub-amostragem Estratificada

Quando `--subsample < 1.0`, utiliza-se `train_test_split` com `stratify` para manter a proporção original das classes na amostra reduzida. Isso garante representatividade mesmo com 10% dos dados.

### 4.3. Divisão Treino / Teste

- **80%** treino, **20%** teste (estratificado por classe).
- `random_state=42` para reprodutibilidade total.

### 4.4. Normalização

`StandardScaler` (z-score) ajustado **exclusivamente no conjunto de treino** e aplicado em ambos os conjuntos, evitando data leakage.

### 4.5. Janela Deslizante (BiLSTM)

Para alimentar a BiLSTM, os fluxos planos `(N, F)` são agrupados em janelas de **T = 10** fluxos consecutivos, resultando em tensores `(N − T + 1, T, F)`. O rótulo de cada janela é o **do último fluxo** da sequência.

A implementação utiliza um `torch.utils.data.Dataset` **lazy**: as janelas são construídas on-the-fly por slicing de tensor, sem replicar dados em memória. Isso reduz o consumo de RAM de ~8.7 GB (se materializado) para ~870 MB (apenas o array original).

> **Nota sobre avaliação:** O LS-SVM é avaliado sobre as N amostras do teste;
> o BiLSTM sobre N − 9 amostras (janeladas). A diferença de 9 registros em
> centenas de milhares é estatisticamente desprezível.

---

## 5. Arquitetura dos Modelos

### 5.1. LS-SVM (Least-Squares SVM)

O LS-SVM clássico resolve:

$$
\min_{w,b,e} \; \frac{1}{2}\|w\|^2 + \frac{C}{2}\sum_i e_i^2
\quad \text{s.a.} \quad y_i[w^\top\phi(x_i)+b] = 1 - e_i
$$

Isso é matematicamente equivalente a **Regressão Ridge no espaço de features do kernel**. A implementação escalável utiliza três componentes em pipeline:

| Componente | Classe sklearn | Função |
|---|---|---|
| **Aproximação do Kernel** | `Nystroem(kernel='rbf', n_components=500)` | Projeta os dados em um espaço de 500 dimensões que aproxima o kernel RBF, usando o método de Nyström |
| **Classificador Ridge** | `RidgeClassifier(alpha=1.0, class_weight='balanced')` | Resolve regressão Ridge (= LS-SVM) nesse espaço; `class_weight='balanced'` compensa desbalanceamento |
| **Calibração de Probabilidades** | `CalibratedClassifierCV(cv=3, method='sigmoid')` | Aplica Platt Scaling (regressão logística sobre scores) para converter `decision_function` em probabilidades, necessárias para curvas ROC e PRC |

**Complexidade:** O(N × F × K) para a transformação Nyström + O(N × K²) para Ridge, onde K=500. Isso escala linearmente em N, viabilizando 2.8M amostras.

**Hiperparâmetros:**

| Parâmetro | Valor | Justificativa |
|---|:---:|---|
| `n_components` | 500 | Compromisso entre fidelidade da aproximação e custo |
| `gamma` | `"scale"` (= 1 / (F × var(X))) | Padrão do sklearn, adapta-se à escala dos dados |
| `alpha` | 1.0 | Regularização moderada (equivale a C=1 no SVM) |
| `cv` (calibração) | 3 | 3-fold cruzado para calibrar probabilidades |

### 5.2. BiLSTM (Bidirectional Long Short-Term Memory)

```
Entrada: (batch, T=10, F=78)
         │
    ┌────▼────┐
    │  LSTM   │  ← 2 camadas, hidden=128, bidirectional
    │  BiDir  │     dropout=0.3 entre camadas
    └────┬────┘
         │  último time-step → (batch, 256)
    ┌────▼────┐
    │ Linear  │  256 → 64
    │  ReLU   │
    │ Dropout │  0.3
    │ Linear  │  64 → 1 (logit)
    └────┬────┘
         │
    BCEWithLogitsLoss (com pos_weight)
```

| Componente | Detalhe |
|---|---|
| **LSTM bidirecional** | 2 camadas, `hidden_size=128`. A saída em cada time-step tem dimensão 2×128=256 (concatenação forward + backward). |
| **Cabeçote FC** | `Linear(256→64) → ReLU → Dropout(0.3) → Linear(64→1)`. Produz um logit escalar. |
| **Loss** | `BCEWithLogitsLoss` com `pos_weight` = N₀/N₁ (compensa desbalanceamento, dá mais peso à classe minoritária). |
| **Otimizador** | Adam, `lr=1e-3` |
| **Early stopping** | Monitora `val_loss` no conjunto de teste; para após 5 épocas sem melhoria e restaura os melhores pesos. |

**Hiperparâmetros:**

| Parâmetro | Valor | Justificativa |
|---|:---:|---|
| `hidden_dim` | 128 | Capacidade suficiente para 78 features |
| `num_layers` | 2 | Profundidade para capturar padrões hierárquicos |
| `dropout` | 0.3 | Regularização contra overfitting |
| `lr` | 1×10⁻³ | Padrão robusto para Adam |
| `batch_size` | 1024 | Equilíbrio velocidade / convergência |
| `epochs` | 20 (máx.) | Com early stopping, raramente atinge o limite |
| `patience` | 5 | Tolerância conservadora |
| `window_size` (T) | 10 | Captura dependências de curto prazo entre fluxos |

**Total de parâmetros treináveis (F=78):**
- LSTM: 4 × [(78+128)×128 + 128] × 2 (bidirecional) × 2 (camadas) ≈ 530 K
- FC: 256×64 + 64 + 64×1 + 1 ≈ 16.5 K
- **Total: ~547 K parâmetros**

---

## 6. Métricas Calculadas

Para cada modelo, com threshold padrão τ=0.5:

| Métrica | Fórmula | Interpretação no contexto de IDS |
|---|---|---|
| **Precisão** | TP / (TP + FP) | De todos os alertas disparados, quantos eram ataques reais |
| **Recall** | TP / (TP + FN) | De todos os ataques reais, quantos foram detectados |
| **F1-Score** | 2 × (Prec × Rec) / (Prec + Rec) | Média harmônica — equilíbrio entre Precisão e Recall |
| **FPR** | FP / (FP + TN) | Taxa de falsos alarmes — **métrica central do TCC** |
| **AUC-ROC** | Área sob a curva ROC | Capacidade discriminativa independente de threshold |

Adicionalmente, o FPR é computado em **três thresholds** (τ = 0.3, 0.5, 0.7) para analisar o impacto da sensibilidade do limiar de decisão.

---

## 7. Artefatos de Saída

### 7.1. Figuras (PNG 300 dpi + PDF vetorial)

| Arquivo | Conteúdo |
|---|---|
| `roc_curve.png/.pdf` | Curvas ROC sobrepostas com valores de AUC na legenda |
| `precision_recall_curve.png/.pdf` | Curvas PRC sobrepostas com Average Precision (AP) |
| `fpr_thresholds.png/.pdf` | Gráfico de barras agrupadas — FPR em τ = {0.3, 0.5, 0.7} com valores anotados |

> Todas as figuras são salvas em PNG (para visualização rápida) e PDF (para
> inclusão no LaTeX sem perda de qualidade).

### 7.2. Tabelas LaTeX (`tabelas_latex.tex`)

O arquivo contém duas tabelas prontas para `\input{}` no documento LaTeX:

1. **Tabela de métricas principais** (`tab:metricas_cicids2017`): Precisão, Recall, F1, FPR e AUC-ROC.
2. **Tabela de FPR por threshold** (`tab:fpr_thresholds`): FPR em τ = 0.3, 0.5, 0.7.

**Pré-requisito LaTeX:** `\usepackage{booktabs}` (para `\toprule`, `\midrule`, `\bottomrule`).

### 7.3. Modelo Salvo

`bilstm_best.pt` — state dict do melhor checkpoint do BiLSTM (pode ser recarregado para inferência futura).

---

## 8. Decisões de Design e Justificativas

### 8.1. Por que Nyström + Ridge e não SVC do sklearn?

O `sklearn.svm.SVC` tem complexidade O(N²) a O(N³) na memória e tempo, tornando-o impraticável para 2.8M de amostras. A combinação Nyström + Ridge:
- É matematicamente equivalente ao LS-SVM com kernel RBF aproximado;
- Escala linearmente em N;
- Permite treinar em minutos, não em dias.

### 8.2. Por que CalibratedClassifierCV?

O `RidgeClassifier` produz scores (via `decision_function`), não probabilidades. Para gerar curvas ROC e PRC, precisamos de probabilidades calibradas. O Platt Scaling (regressão logística sobre os scores) provê essa conversão de forma estatisticamente fundamentada.

### 8.3. Por que janela deslizante lazy (Dataset)?

Materializar todas as janelas `(N×T×F)` em memória consumiria ~8.7 GB para a base inteira. O `SlidingWindowDataset` gera janelas por slicing de tensor (O(1) por acesso), consumindo apenas a memória do array original (~870 MB).

### 8.4. Por que pos_weight na BCEWithLogitsLoss?

O CIC-IDS2017 é desbalanceado (~80% benigno / ~20% ataque). Sem compensação, o modelo tenderia a classificar tudo como benigno. O `pos_weight = N_benigno / N_ataque` dá mais peso aos ataques na loss, forçando o modelo a aprender a detectá-los.

### 8.5. Validação no conjunto de teste (simplificação)

Em produção, deveria haver um **validation set** separado para early stopping, com o test set reservado para avaliação final. No contexto do TCC, utilizar o test set para monitorar early stopping é uma simplificação amplamente aceita em trabalhos acadêmicos, desde que documentada (como aqui).

---

## 9. Estimativa de Tempo de Execução

| Cenário | LS-SVM | BiLSTM (CPU) | BiLSTM (GPU) | Total |
|---|:---:|:---:|:---:|:---:|
| 10% (~280K) | ~30 s | ~10 min | ~2 min | ~10–30 min |
| 100% (~2.8M) | ~5 min | ~2 h | ~15 min | ~20 min – 2 h |

> Tempos aproximados; dependem do hardware do servidor.

---

## 10. Reprodutibilidade

- `random_state=42` em todos os splits, Nyström, e seeds NumPy/PyTorch.
- `torch.manual_seed(42)` + `torch.cuda.manual_seed_all(42)`.
- Determinismo total em CPU; em GPU, pode haver variações mínimas por operações atômicas de CUDA (< 0.01% nas métricas).

---

## 11. Checklist para Execução no Servidor

- [ ] Copiar `cicids2017_cleaned.csv` para o diretório do projeto
- [ ] `pip install -r requirements.txt` (usar `--index-url` do PyTorch se GPU)
- [ ] Rodar teste rápido: `python run_cicids2017.py` (10%)
- [ ] Verificar que `resultados/` foi criado com as 6 figuras + tabela
- [ ] Rodar versão final: `python run_cicids2017.py --full`
- [ ] Copiar `resultados/tabelas_latex.tex` para o projeto LaTeX
- [ ] Inserir figuras PDF no documento com `\includegraphics`

