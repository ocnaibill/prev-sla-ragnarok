# RAGnarok: SLA Prediction & Triage System 

Este repositório contém a solução de Machine Learning desenvolvida para o Problema 3 do Service Desk da RAGnarok. O objetivo do projeto é apoiar a gestão operacional por meio da previsão do tempo de resolução de chamados e da classificação do risco de estouro de SLA no momento da abertura do ticket.

## 📂 Estrutura do Projeto

* `data/raw/`: Dados brutos originais.
* `data/processed/`: Dados limpos gerados após a etapa de limpeza e engenharia de atributos.
* `notebooks/`: Jupyter Notebooks usados para limpeza, análise exploratória, modelagem e avaliação.
* `src/`: Scripts Python para execução operacional do projeto.

## 🛠️ Como configurar o ambiente

1. Clone o repositório:
   ```bash
   git clone https://github.com/ocnaibill/prev-sla-ragnarok.git
   ```

2. Crie um ambiente virtual:
   ```bash
   python -m venv venv
   ```

3. Ative o ambiente virtual.

   No Windows:
   ```bash
   venv\Scripts\activate
   ```

   No Linux/Mac:
   ```bash
   source venv/bin/activate
   ```

4. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

5. Adicione o arquivo de dados do projeto:

   O arquivo original do banco de dados da ANEEL deve ser colocado manualmente na pasta:

   ```text
   data/raw/
   ```

   O nome do arquivo deve ser exatamente:

   ```text
   dados_aneel.xlsx
   ```

   Portanto, o caminho esperado é:

   ```text
   data/raw/dados_aneel.xlsx
   ```

   Esse arquivo não fica salvo diretamente no GitHub por se tratar de um arquivo de dados externo ao código-fonte do projeto.

## ▶️ Como executar o projeto

A execução principal do projeto está organizada nos notebooks da pasta `notebooks/`.

Execute os arquivos na seguinte ordem:

1. `notebooks/01_limpeza_dados.ipynb`

   Responsável por carregar os dados brutos, realizar a limpeza, criar as variáveis-alvo e gerar o arquivo processado:

   ```text
   data/processed/dataset_limpo.csv
   ```

2. `notebooks/02_analise_exploratoria.ipynb`

   Responsável pela análise exploratória dos dados, incluindo gráficos e verificações sobre volume de chamados, distribuição das classes e comportamento das variáveis.

3. `notebooks/03_modelagem.ipynb`

   Responsável pela criação do pipeline de pré-processamento, treinamento dos modelos de classificação e regressão, cálculo das métricas e análise de erro.

## 🚨 Execução operacional final

Após a geração do arquivo `data/processed/dataset_limpo.csv`, é possível executar o script operacional final:

```bash
python src/alerta_operacional.py
```

Esse script simula o funcionamento do sistema no momento em que um chamado é aberto.

Ele utiliza:

* o arquivo `data/processed/dataset_limpo.csv`;
* o mesmo padrão de pré-processamento usado na modelagem;
* o método `.predict_proba()` do modelo de classificação;
* a regra operacional: se a probabilidade de estourar o SLA for maior que 40%, o sistema emite um alerta para escalonamento da equipe.

Saída esperada:

```text
=== Chamado simulado recebido ===
...

=== Resultado da análise operacional ===
Probabilidade estimada de estourar o SLA: ...
Limiar configurado para alerta: 40%

ALERTA OPERACIONAL
Risco acima de 40%. Recomenda-se escalonar o chamado para acompanhamento da equipe.
```

Como o chamado é selecionado aleatoriamente da base de teste, também pode ocorrer uma saída sem alerta:

```text
=== Chamado simulado recebido ===
...

=== Resultado da análise operacional ===
Probabilidade estimada de estourar o SLA: ...
Limiar configurado para alerta: 40%

Sem alerta operacional
Risco igual ou abaixo de 40%. O chamado pode seguir o fluxo normal de atendimento.
```

Caso o arquivo `data/processed/dataset_limpo.csv` não exista, execute primeiro o notebook `01_limpeza_dados.ipynb` para gerar o dataset processado.