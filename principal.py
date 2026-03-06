import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# 1. Configuração da Página
st.set_page_config(layout="wide", page_title="Central de Ajuda Inteligente", page_icon="🔍")

# --- FUNÇÕES DE CARREGAMENTO (CACHE) ---

@st.cache_resource
def carregar_modelo():
    """Carrega o modelo de embeddings uma única vez na memória."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def carregar_dados():
    """Carrega e valida o arquivo Excel."""
    caminho = "faq.xlsx"
    if not os.path.exists(caminho):
        st.error(f"Erro: O arquivo '{caminho}' não foi encontrado.")
        st.stop()
    
    df = pd.read_excel(caminho)
    colunas_obrigatorias = ['categoria', 'pergunta', 'resposta']
    
    # Validação de colunas [4]
    if not all(col in df.columns for col in colunas_obrigatorias):
        st.error(f"O Excel deve conter as colunas: {colunas_obrigatorias}")
        st.stop()
        
    return df.dropna(subset=['pergunta', 'resposta'])

@st.cache_data
def gerar_base_embeddings(_modelo, perguntas):
    """Gera e armazena os embeddings de todas as perguntas cadastras."""
    return _modelo.encode(perguntas.tolist())

# --- LÓGICA DE BUSCA HÍBRIDA ---

def busca_hibrida(query, df, modelo, embeddings_base):
    # A. Busca Semântica (Peso 0.7)
    query_embedding = modelo.encode([query])
    similaridades = cosine_similarity(query_embedding, embeddings_base)
    
    # B. Busca por Palavra-chave (Peso 0.3)
    # Verifica presença do termo na pergunta ou resposta (case insensitive)
    keyword_score = df.apply(
        lambda x: 1.0 if query.lower() in str(x['pergunta']).lower() or 
                         query.lower() in str(x['resposta']).lower() else 0.0, 
        axis=1
    ).values
    
    # C. Ranking Final
    df['score'] = (similaridades * 0.7) + (keyword_score * 0.3)
    return df.sort_values(by='score', ascending=False).head(5)

# --- INTERFACE DO USUÁRIO ---

def main():
    st.title("🔍 Central de Ajuda")
    st.markdown("Busque por dúvidas ou navegue pelas categorias abaixo.")

    # Inicialização [2, 5]
    modelo = carregar_modelo()
    df = carregar_dados()
    embeddings_base = gerar_base_embeddings(modelo, df['pergunta'])

    # Barra de Pesquisa
    query = st.text_input("", placeholder="Digite sua dúvida aqui (ex: Como recuperar senha?)...")

    if query:
        # Modo busca
        resultados = busca_hibrida(query, df, modelo, embeddings_base)
        
        # Filtro de relevância mínima
        resultados = resultados[resultados['score'] > 0.3]
        
        st.subheader(f"Resultados encontrados: {len(resultados)}")
        
        if not resultados.empty:
            for _, row in resultados.iterrows():
                with st.expander(f"📌 {row['pergunta']}", expanded=True):
                    st.write(row['resposta'])
                    st.caption(f"Relevância: {row['score']:.2%}")
        else:
            st.warning("Nenhum resultado relevante encontrado. Tente outras palavras.")
            
    else:
        # Modo Navegação por Categorias [6]
        categorias = df['categoria'].unique()
        tabs = st.tabs(list(categorias))
        
        for i, cat in enumerate(categorias):
            with tabs[i]:
                df_cat = df[df['categoria'] == cat]
                for _, row in df_cat.iterrows():
                    with st.expander(row['pergunta']):
                        st.write(row['resposta'])

    # Rodapé [7]
    st.sidebar.markdown("---")
    st.sidebar.caption("⭐ Central de Ajuda Inteligente\nPreparada para evolução em RAG/LLM")

if __name__ == "__main__":
    main()
