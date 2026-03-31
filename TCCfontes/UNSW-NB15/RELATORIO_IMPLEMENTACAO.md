# Relatório de Implementação — UNSW-NB15

## Classificação Binária de Tráfego de Rede: LS-SVM × BiLSTM

---

## 1. Visão Geral

O script `main.py` implementa um pipeline completo de Machine Learning para classificação binária (Normal × Ataque) sobre o dataset **UNSW-NB15**, utilizando a divisão oficial de treino e teste fornecida pelos autores do dataset. Dois modelos são treinados e comparados:

| Modelo | Tipo | Abordagem |
|--------|------|-----------|
| **LS-SVM** | Machine Learning clássico | Least Squares Support Vector Machine — opera sobre vetores de features estáticos |
| **BiLSTM** | Deep Learning | Bidirectional Long Short-Term Memory — opera sobre janelas sequenciais de fluxos |

---

## 2. Dataset

O **UNSW-NB15** foi criado pelo *Cyber Range Lab* da *Australian Centre for Cyber Security (ACCS)* na UNSW Canberra. Contém tráfego de rede real misturado com ataques sintéticos gerados pela ferramenta IXIA PerfectStorm.

### 2.1 Arquivos Utilizados

| Arquivo | Amostras | Papel |
|---------|----------|-------|
| `UNSW_NB15_training-set.csv` | ~175.341 | Treino |
| `UNSW_NB15_testing-set.csv` | ~82.332 | Teste |

### 2.2 Estrutura Original

Cada arquivo possui **49 colunas**:

- **`id`** — índice sequencial (descartado).
- **42 features numéricas** — `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, `sttl`, `dttl`, `sload`, `dload`, `sloss`, `dloss`, `sinpkt`, `dinpkt`, `sjit`, `djit`, `swin`, `stcpb`, `dtcpb`, `dwin`, `tcprtt`, `synack`, `ackdat`, `smean`, `dmean`, `trans_depth`, `response_body_len`, `ct_srv_src`, `ct_state_ttl`, `ct_dst_ltm`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `ct_dst_src_ltm`, `is_ftp_login`, `ct_ftp_cmd`, `ct_flw_http_mthd`, `ct_src_ltm`, `ct_srv_dst`, `is_sm_ips_ports`.
- **3 features categóricas** — `proto`, `service`, `state`.
- **`attack_cat`** — categoria do ataque (descartada na classificação binária).
- **`label`** — rótulo binário: `0` = Normal, `1` = Ataque.

### 2.3 Distribuição de Classes (aproximada)

| Conjunto | Normal (0) | Ataque (1) | Total |
|----------|-----------|------------|-------|
| Treino | ~56.000 | ~119.341 | ~175.341 |
| Teste | ~37.000 | ~45.332 | ~82.332 |

---

## 3. Pré-processamento (Etapa 1)

### 3.1 Remoção de Colunas

As colunas `id` e `attack_cat` são descartadas:
- **`id`**: índice artificial sem valor preditivo.
- **`attack_cat`**: rótulo multiclasse — irrelevante para a classificação binária.

### 3.2 Extração do Alvo

A coluna `label` é separada como vetor alvo (`y_train`, `y_test`) e removida do DataFrame de features. Seus valores já são binários (0/1), dispensando mapeamento adicional.

### 3.3 Codificação de Features Categóricas

As três colunas categóricas (`proto`, `service`, `state`) são transformadas em valores inteiros via **LabelEncoder** do scikit-learn:

```python
le = LabelEncoder()
combined = pd.concat([df_train[col].astype(str), df_test[col].astype(str)])
le.fit(combined)
```

O `fit` é realizado sobre a **concatenação de treino e teste** para garantir que todos os rótulos categóricos recebam um código consistente, evitando valores desconhecidos durante a inferência.

### 3.4 Tratamento de Valores Ausentes e Infinitos

```python
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)
```

Valores infinitos são substituídos por `NaN` e, em seguida, todos os `NaN` são preenchidos com a **mediana** da respectiva coluna. A mediana é preferida à média por ser robusta a outliers — característica comum em dados de tráfego de rede.

### 3.5 Normalização

Todas as features são normalizadas com **StandardScaler** (z-score):

$$x' = \frac{x - \mu}{\sigma}$$

- O `fit` é realizado **exclusivamente nos dados de treino**.
- O `transform` é aplicado em treino e teste, prevenindo *data leakage*.

---

## 4. Formatação das Entradas (Etapa 2)

### 4.1 Entrada Estática — LS-SVM

O LS-SVM recebe cada amostra como um **vetor unidimensional** de features:

$$\mathbf{X}_{\text{SVM}} \in \mathbb{R}^{N \times F}$$

Onde *N* é o número de amostras e *F* o número de features (≈42 após pré-processamento).

### 4.2 Entrada Sequencial — BiLSTM (Janela Deslizante)

Para o BiLSTM, os dados são reestruturados em **janelas deslizantes** (*sliding windows*) de tamanho `W = 10`:

$$\mathbf{X}_{\text{LSTM}} \in \mathbb{R}^{(N - W + 1) \times W \times F}$$

Cada janela contém 10 amostras consecutivas, e o **rótulo atribuído é o do último elemento** da janela:

```python
for i in range(n_samples):
    Xw[i] = X[i : i + window_size]
    yw[i] = y[i + window_size - 1]
```

**Justificativa**: A janela deslizante permite ao BiLSTM capturar dependências temporais entre fluxos de rede consecutivos, modelando padrões sequenciais que um classificador estático não consegue explorar.

| Conjunto | Shape LS-SVM | Shape BiLSTM |
|----------|-------------|-------------|
| Treino | (175.341, 42) | (175.332, 10, 42) |
| Teste | (82.332, 42) | (82.323, 10, 42) |

---

## 5. Modelo 1 — LS-SVM (Etapa 3)

### 5.1 Fundamentação

O **Least Squares SVM (LS-SVM)** é uma variante do SVM clássico proposta por Suykens e Vandewalle (1999). Diferente do SVM padrão que utiliza restrições de desigualdade e a função de perda *hinge*, o LS-SVM emprega:

- **Restrições de igualdade** no problema de otimização.
- **Função de perda quadrática** (*squared loss*), resultando em um sistema de equações lineares ao invés de programação quadrática.

### 5.2 Implementação

A implementação utiliza o `LinearSVC` do scikit-learn com `loss='squared_hinge'`, que é a aproximação computacional mais direta do LS-SVM:

```python
lsvm_base = LinearSVC(
    loss="squared_hinge",   # perda quadrática — essência do LS-SVM
    dual=True,
    max_iter=10000,
    random_state=42,
)
```

### 5.3 Calibração de Probabilidades

Como o `LinearSVC` nativamente não produz probabilidades (apenas *decision function*), ele é envolvido por `CalibratedClassifierCV` com validação cruzada de 3 folds:

```python
lsvm = CalibratedClassifierCV(estimator=lsvm_base, cv=3)
```

Essa camada ajusta uma regressão logística (Platt scaling) sobre as saídas do SVM, permitindo a geração de `predict_proba` — essencial para o cálculo das curvas ROC e Precision-Recall.

### 5.4 Hiperparâmetros

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `loss` | `squared_hinge` | Formulação LS-SVM |
| `dual` | `True` | Eficiente quando n_features < n_samples |
| `max_iter` | 10.000 | Garantir convergência no dataset grande |
| `cv` (calibração) | 3 | Balanço entre custo e qualidade da calibração |

---

## 6. Modelo 2 — BiLSTM (Etapa 4)

### 6.1 Fundamentação

O **Bidirectional LSTM** processa a sequência de entrada em duas direções (passado→futuro e futuro→passado), concatenando os estados ocultos de ambas as direções. Isso permite capturar dependências temporais tanto anteriores quanto posteriores ao instante atual, tornando-o particularmente eficaz para detecção de intrusão em séries temporais de fluxos de rede.

### 6.2 Arquitetura

```
┌───────────────────────────────────────────────────┐
│  Input: (batch, 10, 42)                           │
├───────────────────────────────────────────────────┤
│  Bidirectional(LSTM(64, return_sequences=True))   │
│  → saída: (batch, 10, 128)                        │
├───────────────────────────────────────────────────┤
│  Dropout(0.3)                                     │
├───────────────────────────────────────────────────┤
│  Bidirectional(LSTM(32))                          │
│  → saída: (batch, 64)                             │
├───────────────────────────────────────────────────┤
│  Dropout(0.3)                                     │
├───────────────────────────────────────────────────┤
│  Dense(32, activation='relu')                     │
├───────────────────────────────────────────────────┤
│  Dense(1, activation='sigmoid')                   │
│  → saída: probabilidade ∈ [0, 1]                  │
└───────────────────────────────────────────────────┘
```

### 6.3 Detalhamento das Camadas

| # | Camada | Unidades | Saída | Função |
|---|--------|----------|-------|--------|
| 1 | Input | — | (10, 42) | Recebe a janela deslizante |
| 2 | Bidirectional LSTM | 64 | (10, 128) | Extração de features sequenciais bidirecionais (retorna sequência completa) |
| 3 | Dropout | 30% | (10, 128) | Regularização contra overfitting |
| 4 | Bidirectional LSTM | 32 | (64,) | Compressão da sequência em vetor contextual |
| 5 | Dropout | 30% | (64,) | Regularização adicional |
| 6 | Dense | 32 | (32,) | Transformação não-linear (ReLU) |
| 7 | Dense (saída) | 1 | (1,) | Classificação binária (sigmoid) |

### 6.4 Compilação e Treinamento

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| Otimizador | Adam | Convergência rápida com taxa adaptativa |
| Função de perda | Binary Crossentropy | Padrão para classificação binária |
| Métrica de monitoramento | Accuracy | Acompanhamento do desempenho |
| Épocas máximas | 30 | Limite superior controlado pelo EarlyStopping |
| Batch size | 256 | Balanço entre velocidade e estabilidade do gradiente |
| Validação | 10% do treino | Monitoramento da generalização |
| EarlyStopping | `patience=5`, `restore_best_weights=True` | Interrompe o treino ao detectar 5 épocas sem melhora na `val_loss`, restaurando os melhores pesos |

### 6.5 Limiar de Decisão

A saída sigmoid é binarizada com limiar padrão de **0.5**:

```python
y_pred_lstm = (y_prob_lstm >= 0.5).astype(int)
```

---

## 7. Métricas de Avaliação (Etapa 5)

### 7.1 Definições

Todas as métricas são calculadas a partir da **matriz de confusão**:

|  | Predito Positivo | Predito Negativo |
|--|-----------------|-----------------|
| **Real Positivo** | VP (True Positive) | FN (False Negative) |
| **Real Negativo** | FP (False Positive) | VN (True Negative) |

### 7.2 Métricas Computadas

| Métrica | Fórmula | Interpretação |
|---------|---------|---------------|
| **Precisão** | $\frac{VP}{VP + FP}$ | Proporção de alertas corretos dentre os emitidos |
| **Recall** (Sensibilidade) | $\frac{VP}{VP + FN}$ | Proporção de ataques reais detectados |
| **F1-Score** | $2 \times \frac{Precisão \times Recall}{Precisão + Recall}$ | Média harmônica entre Precisão e Recall |
| **FPR** (Taxa de Falso Positivo) | $\frac{FP}{FP + VN}$ | Proporção de tráfego normal classificado erroneamente como ataque |
| **AUC-ROC** | Área sob a curva ROC | Capacidade discriminativa global do modelo |

### 7.3 Observação sobre os Conjuntos de Teste

- **LS-SVM**: avaliado sobre os 82.332 exemplos do teste completo.
- **BiLSTM**: avaliado sobre 82.323 exemplos (82.332 − 10 + 1), devido ao descarte inerente da janela deslizante nas primeiras 9 posições. Essa diferença é marginal (~0.01%) e não impacta a comparabilidade dos resultados.

---

## 8. Visualizações Geradas (Etapa 6)

### 8.1 Curva ROC Comparativa (`roc_unsw_nb15.png`)

Plota a **Taxa de Verdadeiro Positivo (TPR)** contra a **Taxa de Falso Positivo (FPR)** para ambos os modelos, com a diagonal aleatória (AUC = 0.5) como referência. A legenda inclui o valor de AUC-ROC de cada modelo.

**Interpretação**: Quanto mais próxima do canto superior esquerdo, melhor o modelo. Um modelo com AUC-ROC superior tem capacidade discriminativa globalmente melhor.

### 8.2 Curva Precision-Recall (`prc_unsw_nb15.png`)

Plota **Precision × Recall** para ambos os modelos. Especialmente informativa em datasets desbalanceados, onde a curva ROC pode ser excessivamente otimista.

**Interpretação**: Quanto maior a área sob a curva PRC, melhor o modelo mantém alta precisão à medida que o recall aumenta.

### 8.3 Gráfico de Barras — FPR (`fpr_unsw_nb15.png`)

Compara visualmente a **Taxa de Falso Positivo** de cada modelo com valores numéricos anotados sobre as barras.

**Interpretação**: Em sistemas de detecção de intrusão, um FPR baixo é crítico — falsos alarmes geram fadiga de alertas e desperdício de recursos de análise.

---

## 9. Saída LaTeX (Etapa 7)

O script imprime no console uma tabela LaTeX completa, pronta para inclusão direta no documento do TCC:

```latex
\begin{table}[htbp]
\centering
\caption{Resultados da Classificação Binária — UNSW-NB15}
\label{tab:unsw_nb15_results}
\begin{tabular}{lccccc}
\toprule
\textbf{Modelo} & \textbf{Precisão} & \textbf{Recall} & \textbf{F1-Score} & \textbf{FPR} & \textbf{AUC-ROC} \\
\midrule
  LS-SVM & x.xxxx & x.xxxx & x.xxxx & x.xxxx & x.xxxx \\
  BiLSTM & x.xxxx & x.xxxx & x.xxxx & x.xxxx & x.xxxx \\
\bottomrule
\end{tabular}
\end{table}
```

A tabela utiliza o pacote `booktabs` para formatação profissional com `\toprule`, `\midrule` e `\bottomrule`.

---

## 10. Dependências e Reprodutibilidade

### 10.1 Bibliotecas

| Biblioteca | Versão Testada | Finalidade |
|------------|---------------|------------|
| `numpy` | ≥ 1.23 | Operações numéricas e arrays |
| `pandas` | ≥ 2.0 | Manipulação de DataFrames |
| `matplotlib` | ≥ 3.7 | Geração de gráficos |
| `scikit-learn` | ≥ 1.3 | LS-SVM, métricas, pré-processamento |
| `tensorflow` | ≥ 2.15 | BiLSTM (Keras integrado) |

### 10.2 Semente de Reprodutibilidade

```python
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

A semente fixa garante que os resultados sejam reprodutíveis entre execuções na mesma máquina e configuração de hardware.

### 10.3 Artefatos Gerados

```
UNSW-NB15/
├── main.py
├── requirements.txt
├── UNSW_NB15_training-set.csv   ← (fornecido pelo usuário)
├── UNSW_NB15_testing-set.csv    ← (fornecido pelo usuário)
└── figures/
    ├── roc_unsw_nb15.png
    ├── prc_unsw_nb15.png
    └── fpr_unsw_nb15.png
```

---

## 11. Instruções de Execução

### 11.1 Pré-requisitos de Ambiente

| Requisito | Versão mínima recomendada |
|-----------|--------------------------|
| Python | 3.8 (testado em 3.10) |
| pip | ≥ 22 |
| RAM | ≥ 8 GB (LS-SVM consome ~4–6 GB com o dataset completo) |
| Disco | ≥ 2 GB livres (CSVs ≈ 300 MB + modelos + figuras) |
| GPU (opcional) | CUDA 11.8 + cuDNN 8.6 para TensorFlow ≥ 2.15 |

### 11.2 Passos

```bash
# 1. (Opcional) Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 2. Instalar dependências CPU
pip install -r requirements.txt

# 2b. (Alternativa GPU) Instalar TensorFlow com suporte CUDA
pip install tensorflow[and-cuda]   # TF >= 2.15 — inclui CUDA/cuDNN automaticamente

# 3. Colocar os dois CSVs oficiais na mesma pasta de main.py:
#    UNSW_NB15_training-set.csv   (~175 k amostras)
#    UNSW_NB15_testing-set.csv    (~82 k amostras)

# 4. Executar
python main.py
```

O script exibe progresso numerado (`[1/7]` a `[7/7]`) no console, facilitando o acompanhamento de cada etapa do pipeline.

### 11.3 Saídas Esperadas

Ao final da execução bem-sucedida:

- Diretório `figures/` criado contendo **3 imagens PNG** (300 dpi).
- Tabela LaTeX impressa diretamente no console — copie e cole no documento `.tex`.
- Mensagem final: `Concluído! Figuras salvas em: <caminho absoluto>/figures`

---

## 12. Checklist para Execução no Servidor

Use esta lista para garantir que nada foi esquecido ao transferir o projeto para outro ambiente.

### 12.1 Preparação dos Arquivos

- [ ] Copiar `UNSW_NB15_training-set.csv` para o diretório raiz do projeto (mesma pasta que `main.py`)
- [ ] Copiar `UNSW_NB15_testing-set.csv` para o diretório raiz do projeto
- [ ] Confirmar tamanhos dos arquivos:
  - `UNSW_NB15_training-set.csv` → ≈ 175.341 linhas (+ cabeçalho), ≈ 50 colunas
  - `UNSW_NB15_testing-set.csv`  → ≈ 82.332 linhas (+ cabeçalho), ≈ 50 colunas

### 12.2 Configuração do Ambiente Python

- [ ] Verificar versão do Python: `python --version` (deve ser ≥ 3.8)
- [ ] Criar ambiente virtual (recomendado): `python -m venv .venv` e ativar
- [ ] Instalar dependências: `pip install -r requirements.txt`
- [ ] (Se GPU disponível) Instalar TensorFlow com CUDA: `pip install tensorflow[and-cuda]`
- [ ] Verificar instalação do TensorFlow: `python -c "import tensorflow as tf; print(tf.__version__)"`
- [ ] (Opcional) Confirmar que a GPU é reconhecida: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

### 12.3 Execução e Verificação

- [ ] Executar o script completo: `python main.py`
- [ ] Confirmar que o console exibiu as etapas `[1/7]` até `[7/7]` sem erros
- [ ] Verificar que o diretório `figures/` foi criado com **exatamente 3 arquivos**:
  - [ ] `figures/roc_unsw_nb15.png`
  - [ ] `figures/prc_unsw_nb15.png`
  - [ ] `figures/fpr_unsw_nb15.png`
- [ ] Confirmar que a tabela LaTeX foi impressa no console ao final da etapa `[7/7]`

### 12.4 Integração com o Documento LaTeX

- [ ] Copiar o bloco `\begin{table}...\end{table}` impresso no console para o arquivo `.tex` desejado
- [ ] Certificar que o preâmbulo do documento inclui `\usepackage{booktabs}` (necessário para `\toprule`, `\midrule`, `\bottomrule`)
- [ ] Copiar os 3 PNGs para a pasta de figuras do projeto LaTeX (ex.: `figuras/`)
- [ ] Inserir as figuras no documento com, por exemplo:

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.75\textwidth]{figuras/roc_unsw_nb15}
  \caption{Curva ROC — UNSW-NB15 (LS-SVM × BiLSTM)}
  \label{fig:roc_unsw_nb15}
\end{figure}
```

### 12.5 Solução de Problemas Comuns

| Sintoma | Causa provável | Solução |
|---------|---------------|---------|
| `FileNotFoundError: UNSW_NB15_training-set.csv` | CSV ausente ou nome errado | Verificar nome exato do arquivo e diretório de execução |
| `AssertionError: Coluna 'label' não encontrada!` | CSV com cabeçalho diferente | Confirmar que é o arquivo oficial UNSW-NB15 |
| `OOM` / `MemoryError` durante LS-SVM | RAM insuficiente | Usar máquina com ≥ 8 GB de RAM ou reduzir `cv=2` no `CalibratedClassifierCV` |
| TensorFlow não detecta GPU | Driver/CUDA incompatível | Verificar compatibilidade em [tensorflow.org/install/pip](https://www.tensorflow.org/install/pip) |
| BiLSTM muito lento (CPU) | Sem GPU | Esperado; o treinamento pode levar 15–40 min em CPU dependendo do hardware |

---

