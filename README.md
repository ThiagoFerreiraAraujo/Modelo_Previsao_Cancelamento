# 📡 Modelo de Previsão de Cancelamento — Operadora de TV a Cabo

Modelo de Machine Learning para prever o cancelamento de clientes (churn) de uma operadora de TV a cabo, a partir de dados cadastrais, financeiros e de uso do serviço.

## 🎯 Objetivo

Identificar, com base no histórico de contratos e pagamentos, quais clientes têm maior probabilidade de cancelar o serviço, permitindo que a operadora antecipe ações de retenção.

## 🗂️ Sobre os dados

- Base original com **448.447 registros** e **24 variáveis**, incluindo dados de aquisição do cliente (forma de aquisição, idade, sexo, quantidade de filhos), dados contratuais (data de aquisição, cancelamento, duração do contrato, produto, plano) e dados financeiros (parcelas pagas, vencidas, pagas em atraso, valores de mensalidade e acordos de pagamento).
- Variável alvo: `SITUACAO` (cliente **Ativo** x **Cancelado**), convertida em `COD_SITUACAO` para o treinamento.

## 🔎 Etapas do projeto

1. **Análise Exploratória de Dados (EDA)**
   - Análise das variáveis categóricas (`FORMA_AQUISICAO`, `SEXO`, `DURACAO_CONTRATO`, `NOME_PRODUTO`, `SITUACAO`) e numéricas.
   - Identificação de desbalanceamento entre clientes ativos e cancelados.

2. **Tratamento de Dados**
   - Remoção de outliers (ex.: registros com `QT_FILHOS` acima de 2).
   - Tratamento de valores nulos (imputação da mediana em `QT_FILHOS`).
   - Correção de inconsistências: quantidades de parcelas pagas/pagas em dia maiores que a duração do contrato foram ajustadas para o limite do contrato.
   - Conversão de `DURACAO_CONTRATO` (texto) para valores numéricos (12, 24, 36 meses).

3. **Engenharia de Atributos**
   - Criação da variável `NIVEL_PAGAMENTO` (RUIM, MEDIO, BOM, OTIMO), categorizando o cliente conforme a quantidade de parcelas pagas.

4. **Codificação e Balanceamento**
   - `LabelEncoder` aplicado às variáveis categóricas (`SEXO`, `FORMA_AQUISICAO`, `NOME_PRODUTO`, etc.).
   - Balanceamento da variável alvo com **SMOTE** (Synthetic Minority Over-sampling Technique), já que havia muito mais clientes ativos do que cancelados.

5. **Padronização e Divisão dos Dados**
   - Padronização das variáveis com `StandardScaler`.
   - Divisão em treino e teste (70% / 30%).

6. **Modelagem**
   - Algoritmo utilizado: **K-Nearest Neighbors (KNN)**.
   - Teste de diferentes valores de `k` (3 a 9) para encontrar o de melhor desempenho.
   - Seleção do modelo final com o `k` de maior acurácia.

## 📊 Resultado

- **Acurácia do modelo final: ~97,7%**

## 🛠️ Tecnologias e bibliotecas

- Python
- Pandas / NumPy — manipulação e tratamento dos dados
- Matplotlib / Seaborn — visualização de dados
- Scikit-learn — pré-processamento, modelagem (KNN) e avaliação (acurácia)
- Imbalanced-learn (SMOTE) — balanceamento da variável alvo

## 📁 Estrutura

```
├── ModeloPrevisaoCancelamento.ipynb   # Notebook com todo o desenvolvimento do projeto
├── dados.csv                          # Base de dados utilizada (não incluída no repositório)
└── README.md
```

## ▶️ Como executar

1. Clone este repositório.
2. Instale as dependências:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
   ```
3. Coloque o arquivo `dados.csv` na raiz do projeto.
4. Execute o notebook `ModeloPrevisaoCancelamento.ipynb` em um Jupyter Notebook/JupyterLab.

## 🚀 Possíveis melhorias futuras

- Testar outros algoritmos de classificação (Random Forest, XGBoost, Regressão Logística) e comparar desempenho.
- Avaliar o modelo com métricas adicionais além da acurácia (precisão, recall, F1-score e matriz de confusão), já que se trata de um problema de classificação com classes originalmente desbalanceadas.
- Realizar tuning de hiperparâmetros com validação cruzada.

---

*Projeto desenvolvido para fins de estudo e prática de Machine Learning.*
