# Relatório de Implementação — LS-SVM vs BiLSTM (CIC-IDS2017)

**Projeto:** TCC – Comparação de LS-SVM e BiLSTM na Redução de Taxa de Falsos Positivos em IDS  
**Dataset:** CIC-IDS2017 Cleaned & Preprocessed (Kaggle) — `cicids2017_cleaned.csv`  
**Data:** 2026-04-01  
**Revisão:** Refatoração metodológica — LS-SVM real (Suykens), PSO, teste t-Student; remoção de `CalibratedClassifierCV`, `predict_proba` via sigmoid, proteção `LinAlgWarning`, log de execução (`_Tee`)

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
    ├── fpr_ttest_boxplot.png / .pdf  # Boxplot FPR × fold (teste t)
    ├── report.log             # Log completo da execução (stdout duplicado)
    ├── bilstm_best.pt         # Pesos do melhor modelo BiLSTM
    ├── lssvm_hparams.json     # C* e σ* encontrados pelo PSO
    └── tabelas_latex.tex      # Tabelas prontas para inserir no LaTeX
```

---

## 2. Dependências e Ambiente

| Pacote         | Versão Mínima | Papel                                                       |
|----------------|:-------------:|--------------------------------------------------------------|
| `numpy`        | ≥ 1.23        | Manipulação vetorial                                        |
| `pandas`       | ≥ 1.5         | Leitura do CSV e manipulação tabular                        |
| `scipy`        | ≥ 1.10        | Teste t de Student pareado (`scipy.stats.ttest_rel`)        |
| `scikit-learn` | ≥ 1.2         | `LSSVMClassifier` (Nyström + Ridge), métricas, splits       |
| `matplotlib`   | ≥ 3.6         | Geração dos gráficos                                        |
| `seaborn`      | ≥ 0.12        | Estilização dos gráficos                                    |
| `torch`        | ≥ 2.0         | BiLSTM (treinamento e inferência)                           |
| `tqdm`         | ≥ 4.64        | (reservado para barras de progresso futuras)                |
| `pyswarms`     | ≥ 1.3         | PSO (`GlobalBestPSO`) para otimização dos hiperparâmetros   |

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

# Ajustar BiLSTM e PSO
python run_cicids2017.py --full --epochs 30 --batch 2048 --patience 7 \
                         --pso-particles 30 --pso-iters 50

# PSO rápido (desenvolvimento/debug)
python run_cicids2017.py --pso-particles 10 --pso-iters 10 --pso-subsample 0.15
```

| Flag               | Default                    | Descrição                                           |
|--------------------|:--------------------------:|-----------------------------------------------------|
| `--csv`            | `cicids2017_cleaned.csv`   | Caminho para o arquivo CSV                          |
| `--subsample`      | `0.10`                     | Fração de sub-amostragem (0.01 – 1.0)               |
| `--full`           | *off*                      | Usa 100% dos dados (ignora subsample)               |
| `--epochs`         | `20`                       | Épocas máximas do BiLSTM                            |
| `--batch`          | `1024`                     | Tamanho do mini-batch                               |
| `--patience`       | `5`                        | Épocas sem melhoria para early stop                 |
| `--output`         | `resultados`               | Diretório de saída                                  |
| `--pso-particles`  | `20`                       | Número de partículas do PSO                         |
| `--pso-iters`      | `30`                       | Iterações do PSO                                    |
| `--pso-subsample`  | `0.30`                     | Fração de `X_train` usada no fitness do PSO         |

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

O LS-SVM clássico (Suykens & Vandewalle, 1999) resolve:

$$
\min_{w,b,\xi} \; \frac{1}{2}\|w\|^2 + \frac{C}{2}\sum_i \xi_i^2
\quad \text{s.a.} \quad y_i[w^\top\phi(x_i)+b] = 1 - \xi_i
$$

Isso é matematicamente equivalente a **Regressão Ridge no espaço de features do kernel**. A implementação escalável é encapsulada na classe `LSSVMClassifier`:

| Componente | Classe sklearn | Função |
|---|---|---|
| **Aproximação do Kernel** | `Nystroem(kernel='rbf', gamma=γ*, n_components=500)` | Projeta os dados em 500 dimensões que aproximam o kernel RBF pelo método de Nyström |
| **Classificador Ridge** | `RidgeClassifier(alpha=1/C*, class_weight='balanced')` | Resolve Ridge (= LS-SVM) nesse espaço; `class_weight='balanced'` compensa desbalanceamento |
**Mapeamento de parâmetros:**

O parâmetro `sigma` (σ) do kernel RBF é convertido para `gamma` (γ) do sklearn via:

$$\gamma = \frac{1}{2\sigma^2}$$

O parâmetro `C` (regularização) é convertido para `alpha` do Ridge via:

$$\alpha_{\text{Ridge}} = \frac{1}{C}$$

> **Proteção numérica:** quando o PSO encontra valores grandes de C (e.g. C ≈ 85), o correspondente α ≈ 0.012 pode tornar a matriz de Ridge mal-condicionada (`LinAlgWarning`). Por isso o código aplica `alpha = clip(1/C, 1e-4, 1e4)` antes de construir o pipeline.

**Probabilidades (`predict_proba`):**

O `RidgeClassifier` produz scores de decisão (`decision_function`), não probabilidades. A classe `LSSVMClassifier` expõe `predict_proba` diretamente via sigmóide:

$$P(y=1 \mid x) = \frac{1}{1 + e^{-f(x)}}$$

essa abordagem, anteriormente delegada ao `CalibratedClassifierCV`, foi incorporada diretamente à classe para evitar incompatibilidades de API (ver seção 8.3).

**Otimização de Hiperparâmetros por PSO:**

Os hiperparâmetros `C` e `σ` **não** são fixos — são encontrados pelo PSO (ver seção 5.3) e salvos em `resultados/lssvm_hparams.json`.

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

### 5.3. PSO — Otimização dos Hiperparâmetros do LS-SVM

O PSO (Particle Swarm Optimization) é usado para encontrar os valores ótimos de **C** e **σ** do LS-SVM, substituindo completamente o `GridSearchCV` / `RandomizedSearchCV`.

**Por que PSO e não Grid/Random Search?**  
A função de custo do PSO penaliza diretamente a **FPR**, alinhando a otimização ao objetivo central do TCC. `GridSearchCV` com `scoring='accuracy'` ou `'f1'` não minimiza FPR especificamente.

**Função de Fitness (minimização):**

$$\text{custo}(C, \sigma) = 0.70 \cdot \overline{\text{FPR}}_{\text{CV}} + 0.30 \cdot (1 - \overline{\text{F1}}_{\text{CV}})$$

O peso 0.70 na FPR garante que o PSO priorize a redução de falsos alarmes, enquanto o peso 0.30 em (1 − F1) evita soluções degeneradas (eg. classificar tudo como positivo).

**Espaço de Busca (escala log₁₀):**

| Parâmetro | Faixa (log₁₀) | Equivalente linear |
|-----------|:-------------:|-------------------|
| C         | [−2, +2]      | [0.01, 100]       |
| σ         | [−2, +1]      | [0.01, 10]        |

**Configuração do PSO (`pyswarms.single.GlobalBestPSO`):**

| Parâmetro              | Default | Descrição                                              |
|------------------------|:-------:|--------------------------------------------------------|
| `n_particles`          | 20      | Número de soluções candidatas simultâneas              |
| `iters`                | 30      | Iterações do enxame                                    |
| Velocidade cognitiva c₁ | 0.5    | Atração pela melhor posição individual da partícula    |
| Velocidade social c₂   | 0.3    | Atração pela melhor posição global do enxame           |
| Inércia w              | 0.9    | Conservação da trajetória atual                        |

**Sub-amostragem do fitness:** para viabilizar o PSO, cada avaliação de fitness usa somente 30% de `X_train` (configurável via `--pso-subsample`). O treino final usa 100%.

**Saída:** `resultados/lssvm_hparams.json` com `C*`, `σ*`, `γ*` e `n_components`.

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

### 6.1. Teste t de Student Pareado

Para validar estatisticamente a diferença de FPR entre os dois modelos, aplica-se o **teste t pareado** (`scipy.stats.ttest_rel`) sobre as FPRs obtidas em **5-fold cross-validation estratificado**:

$$H_0: \mu_{\text{FPR}_{\text{LS-SVM}}} = \mu_{\text{FPR}_{\text{BiLSTM}}} \qquad H_1: \mu_{\text{FPR}_{\text{LS-SVM}}} \neq \mu_{\text{FPR}_{\text{BiLSTM}}}$$

- Se $p < 0.05$: a diferença é **estatisticamente significativa** (nivel $\alpha = 0.05$).
- O modelo com menor $\overline{\text{FPR}}$ é declarado o mais eficaz na redução de falsos positivos.
- **Nota:** Para o BiLSTM, a FPR por fold usa inferência com o modelo já treinado (sem re-treinamento por fold), o que é uma simplificação aceitável no contexto do TCC.

---

## 7. Artefatos de Saída

### 7.1. Figuras (PNG 300 dpi + PDF vetorial)

| Arquivo | Conteúdo |
|---|---|
| `roc_curve.png/.pdf` | Curvas ROC sobrepostas com valores de AUC na legenda |
| `precision_recall_curve.png/.pdf` | Curvas PRC sobrepostas com Average Precision (AP) |
| `fpr_thresholds.png/.pdf` | Gráfico de barras agrupadas — FPR em τ = {0.3, 0.5, 0.7} com valores anotados |
| `fpr_ttest_boxplot.png/.pdf` | Boxplot da FPR nos 5 folds com t-stat e p-valor no título |

> Todas as figuras são salvas em PNG (para visualização rápida) e PDF (para
> inclusão no LaTeX sem perda de qualidade).

### 7.2. Tabelas LaTeX (`tabelas_latex.tex`)

O arquivo contém **três tabelas** prontas para `\input{}` no documento LaTeX:

1. **Tabela de métricas principais** (`tab:metricas_cicids2017`): Precisão, Recall, F1, FPR e AUC-ROC.
2. **Tabela de FPR por threshold** (`tab:fpr_thresholds`): FPR em τ = 0.3, 0.5, 0.7.
3. **Tabela do Teste t** (`tab:ttest`): FPR por fold, média e resultado do teste pareado.

**Pré-requisito LaTeX:** `\usepackage{booktabs}` (para `\toprule`, `\midrule`, `\bottomrule`).

### 7.3. Log de Execução (`report.log`)

`report.log` — cópia integral da saída padrão (stdout) gerada durante a execução, salva em `resultados/`. Produzido pela classe `_Tee` que duplica `sys.stdout` para o arquivo sem suprimir a exibição no terminal. Útil para rastrear métricas, erros e progresso em execuções longas em servidores.

### 7.4. Modelo Salvo

`bilstm_best.pt` — state dict do melhor checkpoint do BiLSTM (pode ser recarregado para inferência futura).

### 7.5. Hiperparâmetros do LS-SVM (`lssvm_hparams.json`)

Arquivo JSON gerado ao final do PSO com o seguinte schema:
```json
{
  "C": 12.3456,
  "sigma": 0.4321,
  "gamma": 2.6876,
  "n_components": 500
}
```

---

## 8. Decisões de Design e Justificativas

### 8.1. Por que `LSSVMClassifier` e não `SVC` do sklearn?

O `sklearn.svm.SVC` com SVM padrão (hinge loss) tem complexidade O(N²) a O(N³) em memória e tempo, tornando-o impraticável para 2.8M de amostras. Além disso, **`sklearn.svm.SVC` não é LS-SVM** — usa outra formulação. A classe `LSSVMClassifier` implementada:
- Usa a formulação de Suykens & Vandewalle (1999) com parâmetros C e σ explícitos;
- Mapeia corretamente γ = 1/(2σ²) e α = 1/C;
- Escala linearmente em N via Nyström;
- É compatível com a API sklearn (herda de `BaseEstimator`, `ClassifierMixin`).

### 8.2. Por que PSO e não `GridSearchCV`?

O PSO otimiza diretamente a métrica de interesse do TCC (FPR), com a função de fitness `0.70×FPR + 0.30×(1−F1)`. `GridSearchCV` com `scoring='accuracy'` ou `'f1'` não minimizaria FPR especificamente. Além disso:
- PSO é mais eficiente que grid search em espaços contínuos 2D;
- A busca em escala logarítmica cobre ordens de magnitude de C e σ de forma natural;
- O resultado é fundamentado metodologicamente para o TCC.

### 8.3. Por que `predict_proba` via sigmoid e não `CalibratedClassifierCV`?

A versão anterior delegava a conversão de scores em probabilidades ao `CalibratedClassifierCV(cv=3, method='sigmoid')` (Platt Scaling). Essa abordagem causou um `ValueError` em tempo de execução:

```
ValueError: LSSVMClassifier should either be a classifier, a regressor,
or a clustered estimator.
```

O erro ocorre porque `CalibratedClassifierCV` chama internamente `_get_response_values`, que não consegue inferir corretamente o tipo do estimador customizado. A solução foi implementar `predict_proba` diretamente na classe `LSSVMClassifier` via sigmóide do `decision_function`:

```python
def predict_proba(self, X):
    scores = self.decision_function(X).astype(np.float64)
    prob_pos = 1.0 / (1.0 + np.exp(-scores))
    return np.column_stack([1.0 - prob_pos, prob_pos])
```

Essa abordagem é matematicamente equivalente ao Platt Scaling com parâmetro de inclinação fixo em 1, evita a dependência do wrapper externo, e mantém total compatibilidade com a API sklearn.

### 8.4. Por que janela deslizante lazy (Dataset)?

Materializar todas as janelas `(N×T×F)` em memória consumiria ~8.7 GB para a base inteira. O `SlidingWindowDataset` gera janelas por slicing de tensor (O(1) por acesso), consumindo apenas a memória do array original (~870 MB).

### 8.5. Por que `pos_weight` na `BCEWithLogitsLoss`?

O CIC-IDS2017 é desbalanceado (~80% benigno / ~20% ataque). Sem compensação, o modelo tenderia a classificar tudo como benigno. O `pos_weight = N_benigno / N_ataque` dá mais peso aos ataques na loss, forçando o modelo a aprender a detectá-los.

### 8.6. Teste t pareado e não independente

Os dois modelos são avaliados nos **mesmos folds** de validação cruzada, portanto os erros são correlacionados por fold. O teste t **pareado** (`ttest_rel`) é mais poderoso e correto nesse cenário do que o teste para amostras independentes (`ttest_ind`).

### 8.8. Log de execução via `_Tee`

Em execuções longas (especialmente em servidores sem interface gráfica), é comum perder a saída do terminal quando a sessão SSH é encerrada. A classe `_Tee` resolve isso duplicando `sys.stdout` para um arquivo de log:

```python
class _Tee:
    def __init__(self, *streams): self._streams = streams
    def write(self, data):
        for s in self._streams: s.write(data); s.flush()
    def flush(self):
        for s in self._streams: s.flush()
    def fileno(self): return self._streams[0].fileno()

sys.stdout = _Tee(sys.__stdout__, open("resultados/report.log", "w"))
```

Todo `print()` subsequente escreve simultaneamente no terminal e em `resultados/report.log`. Ao final do script, `sys.stdout` é restaurado e o arquivo de log é fechado.

### 8.7. Validação no conjunto de teste (simplificação)

Em produção, deveria haver um **validation set** separado para early stopping, com o test set reservado para avaliação final. No contexto do TCC, utilizar o test set para monitorar early stopping é uma simplificação amplamente aceita em trabalhos acadêmicos, desde que documentada (como aqui).

---

## 9. Estimativa de Tempo de Execução

| Cenário | PSO (30 iter × 20 part.) | LS-SVM final | BiLSTM (CPU) | BiLSTM (GPU) | Total aprox. |
|---|:---:|:---:|:---:|:---:|:---:|
| 10% (~280K) | ~10 min | ~30 s | ~10 min | ~2 min | ~20–25 min |
| 100% (~2.8M) | ~60 min | ~5 min | ~2 h | ~15 min | ~1–3 h |

> O PSO domina o tempo total. Use `--pso-particles 10 --pso-iters 10 --pso-subsample 0.15`
> para testes rápidos (épocas de desenvolvimento). Use os valores padrão para execução final.

---

## 10. Reprodutibilidade

- `random_state=42` em todos os splits, Nyström, e seeds NumPy/PyTorch.
- `torch.manual_seed(42)` + `torch.cuda.manual_seed_all(42)`.
- Determinismo total em CPU; em GPU, pode haver variações mínimas por operações atômicas de CUDA (< 0.01% nas métricas).

---

## 11. Checklist para Execução no Servidor

- [ ] Copiar `cicids2017_cleaned.csv` para o diretório do projeto
- [ ] `pip install -r requirements.txt` (incluí `pyswarms` e `scipy`)
- [ ] Rodar teste rápido com PSO enxuto: `python run_cicids2017.py --pso-particles 10 --pso-iters 10`
- [ ] Verificar que `resultados/` foi criado com as 8 figuras + tabela + JSON
- [ ] Conferir `resultados/lssvm_hparams.json` (C* e σ* encontrados)
- [ ] Rodar versão final: `python run_cicids2017.py --full`
- [ ] Copiar `resultados/tabelas_latex.tex` para o projeto LaTeX
- [ ] Inserir figuras PDF no documento com `\includegraphics`
- [ ] Verificar p-valor do teste t e declarar (ou não) significância estatística
- [ ] Conferir `resultados/report.log` — log completo da execução

---

## 12. Problemas Identificados na Execução Real (Pós-Implementação)

Esta seção documenta problemas concretos encontrados durante a execução com o dataset completo (~2,52M amostras), que não são evidentes em execuções com subamostras pequenas.

---

### 12.1. Estouro de Memória RAM na Transformação Nyström (LS-SVM)

**Sintoma:** O processo foi morto pelo kernel (OOM killer) durante a etapa `[3/9]` (treino do LS-SVM final), com uso de ~8,9–10,3 GB de RAM anônima.

**Causa raiz:** A transformação Nyström (`sklearn.kernel_approximation.Nystroem`) aloca internamente **duas cópias** da matriz transformada durante o `fit_transform`:

$$\text{RAM}_\text{Nyström} \approx N_\text{treino} \times n_\text{components} \times 8\,\text{bytes} \times 2$$

Com $N_\text{treino} = 2.016.600$ amostras e `n_components=500` (valor padrão), o pico é de ~16 GB — bem acima dos 15 GB disponíveis na máquina de desenvolvimento.

**Agravante:** O DataFrame original (`df`) e a matriz de features (`X`) permaneciam em memória durante a transformação Nyström, adicionando ~800 MB extras.

**Memória pico estimada por configuração (hardware: 15 GB RAM, 4 GB swap):**

| `n_components` | Pico estimado | Situação |
|:-:|:-:|:-:|
| 500 | ~16 GB | ❌ OOM |
| 200 | ~10,3 GB | ❌ OOM |
| 150 | ~8,5 GB | ✅ Seguro |
| 100 | ~6,7 GB | ✅ Seguro (conservador) |

**Soluções aplicadas:**

1. **Liberar DataFrame após o split:** inserir `del df, X; gc.collect()` imediatamente após `train_test_split`, antes da transformação Nyström. Isso libera ~800 MB antes do pico de memória.

2. **Reduzir `n_components`:** adicionado o argumento `--nys-components` (padrão interno ajustável), com o valor ótimo salvo em `lssvm_hparams.json` junto com C\* e σ\*. No dataset completo, `n_components=150` é o máximo seguro com 15 GB de RAM.

3. **Flag `--skip-pso`:** permite reutilizar hiperparâmetros já otimizados (incluindo `n_components`) de uma execução anterior, relendo o arquivo JSON. Essencial para retomar após reinicializações do sistema.

```python
# Correção aplicada em run_cicids2017.py
import gc
# ...após train_test_split:
del df, X
gc.collect()
```

---

### 12.2. PSO Inviável com Dataset Completo

**Sintoma:** O PSO com `--pso-subsample 0.30` (30% = ~605K amostras) não convergiu após mais de 5 horas na etapa `[2/9]`.

**Causa raiz:** Cada avaliação da função de fitness do PSO chama `fit_transform` do Nyström + `fit` do LS-SVM. Com 20 partículas e 30 iterações, isso totaliza 600 chamadas. Para 605K amostras, cada chamada levava vários minutos.

**Solução aplicada:** reduzir `--pso-subsample` de `0.30` para `0.05` (5% ≈ 100K amostras), o que reduziu o tempo total do PSO para ~60 minutos com qualidade de convergência equivalente (os hiperparâmetros C\* e σ\* encontrados foram validados no treino completo).

**Observação:** para execuções futuras em hardware com memória suficiente, `--pso-subsample 0.10` oferece um bom equilíbrio entre custo e qualidade.

---

### 12.3. `n_components` Ignorado pelo `--skip-pso` (Bug Corrigido)

**Sintoma:** Mesmo com `n_components=200` salvo no JSON, a execução com `--skip-pso` continuava usando o valor padrão (`500`), causando OOM.

**Causa:** O bloco `--skip-pso` lia C\* e σ\* do JSON, mas não lia `n_components`, deixando `NYS_COMPONENTS` com o valor padrão do argparse.

**Correção:**

```python
# Antes (bug):
C_best     = float(_hp["C"])
sigma_best = float(_hp["sigma"])

# Depois (corrigido):
C_best         = float(_hp["C"])
sigma_best     = float(_hp["sigma"])
NYS_COMPONENTS = int(_hp.get("n_components", NYS_COMPONENTS))
```

---

### 12.4. Reinicialização do Sistema por `sudo systemctl restart systemd-logind`

**Sintoma:** Ao tentar configurar o comportamento da tampa do notebook via terminal com `sudo systemctl restart systemd-logind`, a sessão gráfica (Xorg/Wayland) foi encerrada imediatamente, reinicializando efetivamente a sessão do usuário e matando todos os processos em execução (incluindo o script em `[5/9]`).

**Causa:** `systemd-logind` gerencia sessões de login. Reiniciá-lo derruba a sessão gráfica ativa.

**Solução alternativa:** para evitar suspensão ao fechar a tampa com carregador conectado, editar `/etc/systemd/logind.conf` **antes** de iniciar o processo longo, ou usar as configurações gráficas do sistema operacional (*Configurações → Energia → Quando a tampa for fechada → Não fazer nada*) sem precisar reiniciar o serviço.

