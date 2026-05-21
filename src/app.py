import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="RAGnarok - SLA Predictor", page_icon="⚡", layout="centered")

@st.cache_resource
def treinar_modelos():
    caminho_dataset = Path(__file__).resolve().parents[1] / "data" / "processed" / "dataset_limpo.csv"
    df = pd.read_csv(caminho_dataset)
    
    COLUNAS_CATEGORICAS = ["Distribuidora", "Natureza", "Tipologia", "CanalEntrada"]
    X = df.drop(columns=["Protocolo", "estourou_sla", "tempo_resolucao_dias"], errors="ignore")
    y_clf = df["estourou_sla"]
    y_reg = df["tempo_resolucao_dias"]
    
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), COLUNAS_CATEGORICAS)],
        remainder="passthrough"
    )
    
    pipeline_clf = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1))
    ])
    
    pipeline_reg = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1))
    ])
    
    pipeline_clf.fit(X, y_clf)
    pipeline_reg.fit(X, y_reg)
    
    return pipeline_clf, pipeline_reg, df

st.title("⚡ ANEEL: Previsão de Risco de SLA")
st.markdown("Abra um novo chamado simulado para avaliar o risco operacional.")

with st.spinner("Carregando e treinando a Inteligência Artificial..."):
    modelo_clf, modelo_reg, df_base = treinar_modelos()

with st.form("form_chamado"):
    st.subheader("Dados do Chamado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        distribuidora = st.selectbox("Distribuidora", df_base["Distribuidora"].unique())
        natureza = st.selectbox("Natureza", df_base["Natureza"].unique())
        tipologia = st.selectbox("Tipologia", df_base["Tipologia"].unique())
        canal = st.selectbox("Canal de Entrada", df_base["CanalEntrada"].unique())
        
    with col2:
        dia_semana = st.slider("Dia da Semana (0=Seg, 6=Dom)", min_value=0, max_value=6, value=0)
        hora = st.slider("Hora de Abertura", min_value=0, max_value=23, value=9)
        mes = st.slider("Mês de Abertura", min_value=1, max_value=12, value=5)
        
        limiar = st.number_input("Limiar Operacional de Alerta (%)", min_value=10, max_value=90, value=40, step=5)
        
    submit = st.form_submit_button("🔮 Prever Risco Operacional", use_container_width=True)

if submit:
    novo_chamado = pd.DataFrame([{
        "Distribuidora": distribuidora,
        "Natureza": natureza,
        "Tipologia": tipologia,
        "CanalEntrada": canal,
        "dia_semana_abertura": dia_semana,
        "hora_abertura": hora,
        "mes_abertura": mes
    }])
    
    dias_previstos = modelo_reg.predict(novo_chamado)[0]
    prob_atraso = modelo_clf.predict_proba(novo_chamado)[0][1] * 100
    
    st.divider()
    st.subheader("📊 Resultados da Análise IA")
    
    col_res1, col_res2 = st.columns(2)
    col_res1.metric(label="Tempo Estimado de Resolução", value=f"{dias_previstos:.1f} dias")
    col_res2.metric(label="Risco de Estourar SLA", value=f"{prob_atraso:.1f}%")
    
    if prob_atraso >= limiar:
        st.error(f"⚠️ **ALERTA CRÍTICO!** Risco de {prob_atraso:.1f}% superou o limiar de segurança de {limiar}%. Escalonar imediatamente!")
    else:
        st.success(f"✅ **FLUXO NORMAL.** Risco de {prob_atraso:.1f}% está dentro da margem de segurança configurada.")