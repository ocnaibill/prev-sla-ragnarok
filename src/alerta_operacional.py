from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

LIMIAR_ALERTA = 0.40

NOME_ARQUIVO_DATASET = "dataset_limpo.csv"

COLUNA_ALVO_CLASSIFICACAO = "estourou_sla"
COLUNA_ALVO_REGRESSAO = "tempo_resolucao_dias"

COLUNAS_CATEGORICAS = [
    "Distribuidora",
    "Natureza",
    "Tipologia",
    "CanalEntrada",
]

COLUNAS_PARA_REMOVER_SE_EXISTIREM = [
    "Protocolo",              # ID do chamado
    COLUNA_ALVO_CLASSIFICACAO,
    COLUNA_ALVO_REGRESSAO,
]


def obter_raiz_projeto() -> Path:
    return Path(__file__).resolve().parents[1]


def carregar_dataset() -> pd.DataFrame:
    raiz_projeto = obter_raiz_projeto()
    caminho_dataset = raiz_projeto / "data" / "processed" / NOME_ARQUIVO_DATASET

    if not caminho_dataset.exists():
        raise FileNotFoundError(
            "\nArquivo dataset_limpo.csv não encontrado.\n"
            f"Caminho esperado: {caminho_dataset}\n\n"
            "Antes de rodar este script, confirme se o notebook de limpeza gerou "
            "o arquivo data/processed/dataset_limpo.csv."
        )

    return pd.read_csv(caminho_dataset)


def validar_colunas_necessarias(df: pd.DataFrame) -> None:
    colunas_necessarias = COLUNAS_CATEGORICAS + [COLUNA_ALVO_CLASSIFICACAO]

    colunas_faltando = [
        coluna for coluna in colunas_necessarias
        if coluna not in df.columns
    ]

    if colunas_faltando:
        raise ValueError(
            "\nO dataset não possui todas as colunas esperadas pelo alerta operacional.\n"
            f"Colunas faltando: {colunas_faltando}\n\n"
            "Verifique se o arquivo data/processed/dataset_limpo.csv foi gerado "
            "com a mesma estrutura usada no notebook 03_modelagem.ipynb."
        )


def preparar_dados_classificacao(df: pd.DataFrame):
    validar_colunas_necessarias(df)

    colunas_para_remover = [
        coluna for coluna in COLUNAS_PARA_REMOVER_SE_EXISTIREM
        if coluna in df.columns
    ]

    X = df.drop(columns=colunas_para_remover)
    y = df[COLUNA_ALVO_CLASSIFICACAO]

    return X, y


def criar_pipeline_classificacao() -> ImbPipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), COLUNAS_CATEGORICAS)
        ],
        remainder="passthrough"
    )

    pipeline_classificacao = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(
            max_depth=12,
            min_samples_leaf=3,
            random_state=42
        ))
    ])

    return pipeline_classificacao


def treinar_modelo_classificacao(X: pd.DataFrame, y: pd.Series) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    modelo = criar_pipeline_classificacao()
    modelo.fit(X_train, y_train)

    return modelo, X_test, y_test


def obter_indice_classe_atraso(modelo: ImbPipeline) -> int:
    classes = list(modelo.classes_)

    if 1 not in classes:
        raise ValueError(
            "O modelo treinado não encontrou a classe 1 em estourou_sla. "
            "Não é possível calcular o risco de estouro de SLA."
        )

    return classes.index(1)


def escolher_chamado_para_simulacao(modelo: ImbPipeline, X_test: pd.DataFrame) -> pd.DataFrame:
    chamado_simulado = X_test.sample(n=1)

    return chamado_simulado


def calcular_probabilidade_estouro_sla(
    modelo: ImbPipeline,
    chamado: pd.DataFrame
) -> float:
    """Calcula a probabilidade estimada de o chamado estourar o SLA."""
    indice_classe_1 = obter_indice_classe_atraso(modelo)

    probabilidades = modelo.predict_proba(chamado)
    probabilidade_estouro = float(probabilidades[0][indice_classe_1])

    return probabilidade_estouro


def exibir_chamado_simulado(chamado: pd.DataFrame) -> None:
    print("\n=== Chamado simulado recebido ===")
    print("Este chamado foi selecionado aleatoriamente da base de teste para simular um chamado recém-aberto.")

    linha = chamado.iloc[0]

    for coluna in chamado.columns:
        print(f"{coluna}: {linha[coluna]}")


def exibir_resultado_operacional(probabilidade_estouro: float) -> None:
    probabilidade_percentual = probabilidade_estouro * 100
    limiar_percentual = LIMIAR_ALERTA * 100

    print("\n=== Resultado da análise operacional ===")
    print(f"Probabilidade estimada de estourar o SLA: {probabilidade_percentual:.2f}%")
    print(f"Limiar configurado para alerta: {limiar_percentual:.0f}%")

    if probabilidade_estouro > LIMIAR_ALERTA:
        print("\nALERTA OPERACIONAL")
        print(
            "Risco acima de 40%. Recomenda-se escalonar o chamado "
            "para acompanhamento da equipe."
        )
    else:
        print("\nSem alerta operacional")
        print(
            "Risco igual ou abaixo de 40%. O chamado pode seguir "
            "o fluxo normal de atendimento."
        )


def main() -> None:
    print("Carregando dataset limpo...")
    df = carregar_dataset()

    print("Preparando dados de classificação...")
    X, y = preparar_dados_classificacao(df)

    print("Treinando modelo de classificação usado no alerta operacional...")
    modelo, X_test, _ = treinar_modelo_classificacao(X, y)

    print("Selecionando chamado simulado para teste operacional...")
    chamado_simulado = escolher_chamado_para_simulacao(modelo, X_test)

    probabilidade_estouro = calcular_probabilidade_estouro_sla(
        modelo,
        chamado_simulado
    )

    exibir_chamado_simulado(chamado_simulado)
    exibir_resultado_operacional(probabilidade_estouro)


if __name__ == "__main__":
    main()
