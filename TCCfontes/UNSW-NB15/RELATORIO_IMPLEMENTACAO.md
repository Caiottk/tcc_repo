# Relatório de Implementação — UNSW-NB15

## Classificação Binária de Tráfego de Rede: LS-SVM × BiLSTM

**Projeto:** TCC – Comparação de LS-SVM e BiLSTM na Redução de Taxa de Falsos Positivos em IDS  
**Dataset:** UNSW-NB15 (divisão oficial de treino/teste fornecida pelos autores)  
**Data:** 2026-04-01  
**Revisão:** Refatoração metodológica — LS-SVM real (Suykens), PSO, teste t-Student; remoção de `CalibratedClassifierCV`, `predict_proba` via sigmoid, proteção `LinAlgWarning`, log de execução (`_Tee`)

---

## 1. Estrutura do Projeto

```
UNSW-NB15/
├── main.py                            # Script principal (único arquivo)
├── requirements.txt                   # Dependências Python
├── UNSW_NB15_training-set.csv         # ← colocar o CSV aqui (não versionado)
├── UNSW_NB15_testing-set.csv          # ← colocar o CSV aqui (não versionado)
└── resultados/                        # Gerado automaticamente na execução
    ├── report.log                     # Log completo da execução (stdout duplicado)
    ├── roc_unsw_nb15.png / .pdf
    ├── prc_unsw_nb15.png / .pdf
    ├── fpr_unsw_nb15.png / .pdf
    ├── fpr_ttest_boxplot.png / .pdf   # Boxplot FPR × fold (teste t)
    ├── lssvm_hparams.json             # C* e σ* encontrados pelo PSO
    └── tabelas_latex.tex              # Tabelas prontas para inserir no LaTeX
```

---

## 2. Dependências e Ambiente

```
numpy>=1.23
pandas>=1.5
scipy>=1.10
scikit-learn>=1.2
matplotlib>=3.6
tensorflow>=2.15
pyswarms>=1.3
```

```bash
pip install -r requirements.txt
```

---

## 3. Dataset — UNSW-NB15

O **UNSW-NB15** foi criado pelo *Cyber Range Lab* da *Australian Centre for Cyber Security (ACCS)* na UNSW Canberra. Contém tráfego de rede real misturado com ataques sintéticos gerados pela ferramenta IXIA PerfectStorm.

### 3.1 Arquivos Utilizados

| Arquivo | Amostras | Papel |
|---------|----------|-------|
| `UNSW_NB15_training-set.csv` | ~175.341 | Treino |
| `UNSW_NB15_testing-set.csv` | ~82.332 | Teste |

### 3.2 Estrutura Original

Cada arquivo possui **49 colunas**:

- **`id`** — índice sequencial (descartado).
- **42 features numéricas** — `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, `sttl`, `dttl`, `sload`, `dload`, `sloss`, `dloss`, `sinpkt`, `dinpkt`, `sjit`, `djit`, `swin`, `stcpb`, `dtcpb`, `dwin`, `tcprtt`, `synack`, `ackdat`, `smean`, `dmean`, `trans_depth`, `response_body_len`, `ct_srv_src`, `ct_state_ttl`, `ct_dst_ltm`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `ct_dst_src_ltm`, `is_ftp_login`, `ct_ftp_cmd`, `ct_flw_http_mthd`, `ct_src_ltm`, `ct_srv_dst`, `is_sm_ips_ports`.
- **3 features categóricas** — `proto`, `service`, `state`.
- **`attack_cat`** — categoria do ataque (descartada na classificação binária).
- **`label`** — rótulo binário: `0` = Normal, `1` = Ataque.

### 3.3 Distribuição de Classes (aproximada)

| Conjunto | Normal (0) | Ataque (1) | Total |
|----------|-----------|------------|-------|
| Treino | ~56.000 | ~119.341 | ~175.341 |
| Teste | ~37.000 | ~45.332 | ~82.332 |

---

## 4. Pré-processamento

### 4.1 Remoção de Colunas

As colunas `id` e `attack_cat` são descartadas:
- **`id`**: índice artificial sem valor preditivo.
- **`attack_cat`**: rótulo multiclasse — irrelevante para a classificação binária.

### 4.2 Extração do Alvo

A coluna `label` é separada como vetor alvo (`y_train`, `y_test`) e removida do DataFrame de features. Seus valores já são binários (0/1), dispensando mapeamento adicional.

### 4.3 Codificação de Features Categóricas

As três colunas categóricas (`proto`, `service`, `state`) são transformadas em valores inteiros via **LabelEncoder** do scikit-learn. O `fit` é realizado sobre a **concatenação de treino e teste** para garantir consistência de códigos.

### 4.4 Tratamento de Valores Ausentes e Infinitos

```python
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)
```

Valores infinitos são substituídos por `NaN` e preenchidos com a **mediana** da respectiva coluna (robusta a outliers).

### 4.5 Normalização

Todas as features são normalizadas com **StandardScaler** (z-score). O `fit` é realizado **exclusivamente nos dados de treino** para evitar *data leakage*.

---

## 5. Modelo 1 — LS-SVM

### 5.1 Fundamentação

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

O `RidgeClassifier` produz scores de decisão (`decision_function`), não probabilidades. A classe `LSSVMClassifier` expõe `predict_proba` diretamente via sigmoide:

$$P(y=1 \mid x) = \frac{1}{1 + e^{-f(x)}}$$

Essa abordagem substitui o anterior `CalibratedClassifierCV` para evitar incompatibilidades de API (ver seção 8.3).

**Otimização de Hiperparâmetros por PSO:**

Os hiperparâmetros `C` e `σ` **não** são fixos — são encontrados pelo PSO (ver seção 5.3) e salvos em `resultados/lssvm_hparams.json`.

### 5.2 Entrada

O LS-SVM recebe cada amostra como um **vetor unidimensional** de features:

$$\mathbf{X}_{\text{SVM}} \in \mathbb{R}^{N \times F}$$

### 5.3 PSO — Otimização de C e σ

O **Particle Swarm Optimization** (`pyswarms.single.GlobalBestPSO`) minimiza a função de fitness:

$$\text{custo} = 0.70 \times \overline{FPR}_{\text{CV}} + 0.30 \times (1 - \overline{F1}_{\text{CV}})$$

com validação cruzada 3-fold sobre uma amostra aleatória do treino (`--pso-subsample`, default 20%).

| Parâmetro PSO | Default | CLI |
|---|---|---|
| Partículas | 20 | `--pso-particles` |
| Iterações | 30 | `--pso-iters` |
| Subsample | 20% | `--pso-subsample` |

O espaço de busca é log-uniforme: C ∈ [10⁻², 10⁴], σ ∈ [10⁻³, 10³].

---

## 6. Modelo 2 — BiLSTM

### 6.1 Entrada Sequencial (Janela Deslizante)

Para o BiLSTM, os dados são reestruturados em **janelas deslizantes** (*sliding windows*) de tamanho `W = 10`:

$$\mathbf{X}_{\text{LSTM}} \in \mathbb{R}^{(N - W + 1) \times W \times F}$$

O **rótulo atribuído é o do último elemento** da janela. A janela deslizante permite ao BiLSTM capturar dependências temporais entre fluxos de rede consecutivos.

| Conjunto | Shape LS-SVM | Shape BiLSTM |
|----------|-------------|-------------|
| Treino | (175.341, 42) | (175.332, 10, 42) |
| Teste | (82.332, 42) | (82.323, 10, 42) |

### 6.2 Arquitetura

```
Input: (batch, 10, 42)
         │
    ┌────▼────┐
    │ BiLSTM  │  64 unidades, return_sequences=True → (batch, 10, 128)
    └────┬────┘
    Dropout(0.3)
    ┌────▼────┐
    │ BiLSTM  │  32 unidades → (batch, 64)
    └────┬────┘
    Dropout(0.3)
    Dense(32, relu)
    Dense(1, sigmoid) → probabilidade ∈ [0, 1]
```

### 6.3 Compilação e Treinamento

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| Otimizador | Adam | Convergência rápida com taxa adaptativa |
| Função de perda | Binary Crossentropy | Padrão para classificação binária |
| Épocas máximas | 30 | Controlado pelo EarlyStopping |
| Batch size | 256 | Balanço entre velocidade e estabilidade |
| Validação | 10% do treino | Monitoramento da generalização |
| EarlyStopping | `patience=5` | Restaura melhores pesos |
| `class_weight` | `{0: 1.0, 1: N_neg/N_pos}` | Compensa desbalanceamento de classes |

---

## 7. Artefatos de Saída

### 7.1 Figuras (PNG 300 dpi + PDF vetorial)

| Arquivo | Conteúdo |
|---|---|
| `roc_unsw_nb15.png/.pdf` | Curvas ROC sobrepostas com valores de AUC na legenda |
| `prc_unsw_nb15.png/.pdf` | Curvas PRC sobrepostas |
| `fpr_unsw_nb15.png/.pdf` | Gráfico de barras — FPR com valores anotados |
| `fpr_ttest_boxplot.png/.pdf` | Boxplot da FPR nos 5 folds com t-stat e p-valor no título |

> Todas as figuras são salvas em PNG (visualização rápida) e PDF (inclusão no LaTeX sem perda de qualidade).

### 7.2 Log de Execução (`report.log`)

`report.log` — cópia integral da saída padrão (stdout) gerada durante a execução, salva em `resultados/`. Produzido pela classe `_Tee` que duplica `sys.stdout` para o arquivo sem suprimir a exibição no terminal.

### 7.3 Hiperparâmetros do LS-SVM (`lssvm_hparams.json`)

```json
{
  "C": 12.3456,
  "sigma": 0.4321,
  "gamma": 2.6876,
  "n_components": 500
}
```

### 7.4 Tabelas LaTeX (`tabelas_latex.tex`)

O arquivo contém **duas tabelas** prontas para `\input{}` no documento LaTeX:

1. **Tabela de métricas principais** (`tab:unsw_nb15_results`): Precisão, Recall, F1, FPR e AUC-ROC.
2. **Tabela do Teste t** (`tab:ttest_unsw`): FPR por fold, média e p-valor.

**Pré-requisito LaTeX:** `\usepackage{booktabs}`.

---

## 8. Validação Estatística — Teste t-Student Pareado

O script realiza um **5-fold stratified CV** sobre o conjunto unificado (treino + teste) para obter a FPR de cada modelo em 5 partições independentes. Em seguida aplica o teste t **pareado** (`scipy.stats.ttest_rel`):

$$H_0: \mu_{\text{FPR,LS-SVM}} = \mu_{\text{FPR,BiLSTM}}$$

- **Pareado** porque os dois modelos são avaliados nos **mesmos folds** — os erros são correlacionados.
- Resultado declarado como significativo se p < 0.05 (α = 5%).

O boxplot `fpr_ttest_boxplot.png` exibe a distribuição de FPR por modelo com t e p no título.

---

## 9. Decisões de Design e Justificativas

### 9.1 Por que `LSSVMClassifier` e não `LinearSVC`?

O `LinearSVC` com `loss='squared_hinge'` é uma aproximação computacional, mas **não implementa explicitamente** os parâmetros C e σ do LS-SVM de Suykens & Vandewalle. A classe `LSSVMClassifier` implementada:
- Usa a formulação correta com C e σ explícitos;
- Mapeia corretamente γ = 1/(2σ²) e α = clip(1/C, 1e-4, 1e4);
- Escala linearmente em N via Nyström;
- É compatível com a API sklearn (herda de `BaseEstimator`, `ClassifierMixin`).

### 9.2 Por que PSO e não `GridSearchCV`?

O PSO otimiza diretamente a métrica de interesse do TCC (FPR), com `0.70×FPR + 0.30×(1−F1)`. `GridSearchCV` com `scoring='accuracy'` ou `'f1'` não minimizaria FPR especificamente. Além disso, PSO é mais eficiente em espaços contínuos 2D (log C × log σ).

### 9.3 Por que `predict_proba` via sigmoid e não `CalibratedClassifierCV`?

A versão anterior delegava a conversão de scores em probabilidades ao `CalibratedClassifierCV(cv=3, method='sigmoid')` (Platt Scaling). Essa abordagem causou um `ValueError` em tempo de execução:

```
ValueError: LSSVMClassifier should either be a classifier, a regressor,
or a clustered estimator.
```

O erro ocorre porque `CalibratedClassifierCV` chama internamente `_get_response_values`, que não consegue inferir corretamente o tipo do estimador customizado. A solução foi implementar `predict_proba` diretamente na classe `LSSVMClassifier` via sigmoide do `decision_function`:

```python
def predict_proba(self, X):
    scores = self.decision_function(X).astype(np.float64)
    prob_pos = 1.0 / (1.0 + np.exp(-scores))
    return np.column_stack([1.0 - prob_pos, prob_pos])
```

Matematicamente equivalente ao Platt Scaling com inclinação fixo em 1, sem dependência do wrapper externo.

### 9.4 Por que janela deslizante eager (numpy)?

O dataset UNSW-NB15 (~258k amostras) cabe confortavelmente em memória após o windowing (~870 MB). A abordagem eager (materializar todas as janelas em numpy) é mais simples e compatível com a API do Keras (`model.fit` recebe arrays diretamente).

### 9.5 Por que `class_weight` no BiLSTM?

O UNSW-NB15 de treino é desbalanceado (~32% Normal / ~68% Ataque, invertido ao esperado). O `class_weight = {0: 1.0, 1: N_neg/N_pos}` garante que o modelo não ignore a classe minoritária.

### 9.6 Teste t pareado e não independente

Os dois modelos são avaliados nos **mesmos folds** de validação cruzada. O teste t **pareado** (`ttest_rel`) é mais poderoso e correto nesse cenário do que o teste para amostras independentes (`ttest_ind`).

### 9.7 Validação no conjunto de teste (simplificação)

Em produção, deveria haver um **validation set** separado para early stopping. No contexto do TCC, utilizar o test set para monitorar early stopping é uma simplificação amplamente aceita, desde que documentada.

### 9.8 Log de execução via `_Tee`

Em execuções longas em servidores sem interface gráfica, é comum perder a saída do terminal quando a sessão SSH é encerrada. A classe `_Tee` resolve isso duplicando `sys.stdout` para um arquivo de log:

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

---

## 10. Estimativa de Tempo de Execução

| Cenário | PSO (30 iter × 20 part.) | LS-SVM final | BiLSTM (CPU) | BiLSTM (GPU) | Total aprox. |
|---|:---:|:---:|:---:|:---:|:---:|
| Execução completa + t-test | ~30 min | ~1 min | ~15 min/fold × 5 | ~3 min/fold × 5 | ~2–4 h (CPU) |

> O t-test domina o tempo total por treinar o BiLSTM 5× em folds. Use
> `--pso-particles 10 --pso-iters 10 --pso-subsample 0.15` para testes rápidos.

---

## 11. Reprodutibilidade

- `SEED = 42` em todos os splits, Nyström, e seeds NumPy/TensorFlow.
- `np.random.seed(42)` + `tf.random.set_seed(42)`.
- Determinismo total em CPU; em GPU, pode haver variações mínimas por operações atômicas de CUDA (< 0.01% nas métricas).

---

## 12. Checklist para Execução no Servidor

### 12.1 Preparação dos Arquivos

- [ ] Copiar `UNSW_NB15_training-set.csv` para o diretório raiz do projeto
- [ ] Copiar `UNSW_NB15_testing-set.csv` para o diretório raiz do projeto
- [ ] Confirmar tamanhos: treino ≈ 175.341 linhas, teste ≈ 82.332 linhas

### 12.2 Configuração do Ambiente Python

- [ ] Verificar versão do Python: `python --version` (deve ser ≥ 3.8)
- [ ] Criar ambiente virtual (recomendado): `python -m venv .venv` e ativar
- [ ] Instalar dependências: `pip install -r requirements.txt`
- [ ] (Se GPU disponível) Instalar TensorFlow com CUDA: `pip install tensorflow[and-cuda]`

### 12.3 Execução e Verificação

- [ ] Rodar teste rápido: `python main.py --pso-particles 10 --pso-iters 10 --pso-subsample 0.15`
- [ ] Verificar que `resultados/` foi criado com os artefatos esperados
- [ ] Conferir `resultados/lssvm_hparams.json` (C* e σ* encontrados)
- [ ] Rodar versão final: `python main.py`
- [ ] Copiar `resultados/tabelas_latex.tex` para o projeto LaTeX
- [ ] Inserir figuras PDF no documento com `\includegraphics`
- [ ] Verificar p-valor do teste t e declarar (ou não) significância estatística
- [ ] Conferir `resultados/report.log` — log completo da execução

### 12.4 Solução de Problemas Comuns

| Sintoma | Causa provável | Solução |
|---------|---------------|---------|
| `FileNotFoundError: UNSW_NB15_training-set.csv` | CSV ausente ou nome errado | Verificar nome exato do arquivo e diretório de execução |
| `AssertionError: Coluna 'label' não encontrada!` | CSV com cabeçalho diferente | Confirmar que é o arquivo oficial UNSW-NB15 |
| `OOM` / `MemoryError` durante PSO | RAM insuficiente | Reduzir `--pso-subsample` para 0.10 ou menos |
| TensorFlow não detecta GPU | Driver/CUDA incompatível | Verificar compatibilidade em [tensorflow.org/install/pip](https://www.tensorflow.org/install/pip) |
| BiLSTM muito lento (CPU) | Sem GPU | Esperado; treino pode levar 15–40 min por fold em CPU |
