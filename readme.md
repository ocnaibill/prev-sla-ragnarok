
# RAGnarok: SLA Prediction & Triage System 

Este repositório contém a solução de Machine Learning (Problema 3) desenvolvida para o Service Desk da RAGnarok. O objetivo é prever o tempo de resolução de chamados de TI e classificar o risco de estouro de SLA (Service Level Agreement) no momento da abertura do ticket.

## 📂 Estrutura do Projeto

* `data/raw/`: Dados brutos originais (CSVs não processados).
* `data/processed/`: Dados limpos após Feature Engineering. 
* `notebooks/`: Jupyter Notebooks para Provas de Conceito (PoC) e Análise Exploratória (EDA).
* `src/`: Scripts Python contendo o Pipeline Scikit-Learn e os algoritmos em produção.

## 🛠️ Como configurar o ambiente

1. Clone o repositório:
   ```bash
   git clone git@github.com:SEU-USUARIO/ragnarok-sla-prediction.git
   ```
2. Crie um ambiente virtual e ative:
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows use: venv\Scripts\activate
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt