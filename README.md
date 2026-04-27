# TCC: Comparação LS-SVM vs BiLSTM na Redução de Falsos Positivos em IDS

## Visão Geral

Este repositório contém a implementação e análise comparativa de dois modelos de aprendizado de máquina para detecção de intrusões em sistemas de segurança (IDS): **LS-SVM** (Least-Squares Support Vector Machine) e **BiLSTM** (Bidirectional Long Short-Term Memory). O foco principal é avaliar a capacidade de cada modelo em reduzir falsos positivos (FPR) em cenários de detecção de ataques.

## Datasets

- **CIC-IDS2017**: ~2.8 milhões de amostras, altamente desbalanceado (99.9% benigno), 78 features
- **UNSW-NB15**: ~257 mil amostras (175k treino, 82k teste), balanceado, 42 features

## Resultados Principais

| Dataset | Modelo | FPR | p-valor (t-test) | Significância |
|---------|--------|-----|-----------------|---------------|
| CIC-IDS2017 | LS-SVM | 0.0077 | 0.2254 | Não significativa |
| CIC-IDS2017 | BiLSTM | 0.0077 | - | - |
| UNSW-NB15 | LS-SVM | 0.2214 | 0.0069 | Significativa (α=0.05) |
| UNSW-NB15 | BiLSTM | 0.0215 | - | 10× redução FPR |

**Principais Achados:**
- Em datasets balanceados (UNSW-NB15), BiLSTM reduz FPR em ~10× comparado ao LS-SVM
- Em datasets extremos (CIC-IDS2017), ambos modelos atingem FPR mínimo devido ao desbalanceamento
- LS-SVM via Nyström escala bem para grandes datasets sem OOM

## Estrutura do Repositório

```
tcc_repo/
├── TCCfontes/
│   ├── CIC-IDS2017/
│   │   ├── run_cicids2017.py          # Script principal CIC-IDS2017
│   │   ├── cicids2017_cleaned.csv     # Dataset processado
│   │   ├── resultados/                # Outputs: figuras, tabelas, logs
│   │   └── requirements.txt
│   └── UNSW-NB15/
│       ├── main.py                    # Script principal UNSW-NB15
│       ├── UNSW_NB15_*.csv            # Datasets oficiais
│       ├── resultados/                # Outputs similares
│       └── requirements.txt
├── RELATORIO_IMPLEMENTACAO.md         # Documentação técnica detalhada
└── README.md                          # Este arquivo
```

## Como Executar

### Pré-requisitos
```bash
pip install -r TCCfontes/CIC-IDS2017/requirements.txt
# ou
pip install -r TCCfontes/UNSW-NB15/requirements.txt
```

### Execução Rápida (Desenvolvimento)
```bash
# CIC-IDS2017 (10% dos dados, teste rápido)
cd TCCfontes/CIC-IDS2017
python run_cicids2017.py

# UNSW-NB15 (completo, mas com PSO reduzido)
cd ../UNSW-NB15
python main.py --pso-particles 10 --pso-iters 10
```

### Execução Completa (Produção)
```bash
# CIC-IDS2017 completo (~30h PSO)
python run_cicids2017.py --full --pso-particles 20 --pso-iters 30

# UNSW-NB15 completo (~15h PSO)
python main.py --pso-particles 20 --pso-iters 30
```

## Principais Limitações

- **Hardware**: Limitado a CPU-only (15GB RAM), impossibilita n_components > 150 em Nyström
- **BiLSTM**: Contaminação treino-validação devido a janelas deslizantes sobrepostas (9/10 timesteps)
- **Tempo**: PSO completo leva 15-30h dependendo do dataset
- **Escalabilidade**: LS-SVM limitado por memória Nyström, BiLSTM por sequência temporal

## Arquivos de Saída

Cada script gera em `resultados/`:
- `roc_*.{png,pdf}`: Curva ROC comparativa
- `prc_*.{png,pdf}`: Curva Precision-Recall
- `fpr_*.{png,pdf}`: Gráfico de barras FPR
- `fpr_ttest_boxplot.{png,pdf}`: Boxplot FPR por fold (teste t)
- `lssvm_hparams.json`: Hiperparâmetros ótimos encontrados
- `tabelas_latex.tex`: Tabelas formatadas para LaTeX
- `report.log`: Log completo da execução

## Referências Técnicas

- Suykens & Vandewalle (1999): Formulação primal LS-SVM
- Williams & Seeger (2001): Aproximação Nyström para kernels
- Kennedy & Eberhart (1995): Particle Swarm Optimization
- Hochreiter & Schmidhuber (1997): LSTM networks