import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# 1. Configuração da Página
st.set_page_config(layout="wide", page_title="FAQ Inteligente")

# --- CACHE DE DADOS E MODELO ---
@st.cache_resource
def carregar_modelo():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def carregar_dados():
    df = pd.read_excel("faq.xlsx")
    df.columns = df.columns.str.strip()
    return df.dropna(subset=['pergunta', 'resposta'])

@st.cache_data
def gerar_embeddings(_modelo, perguntas):
    return _modelo.encode(perguntas.tolist())

# --- LÓGICA DE BUSCA ---
def busca_hibrida(query, df, modelo, embeddings_base):
    query_embedding = modelo.encode([query])
    # .flatten() corrige o erro de dimensão [ValueError: Length of values (1) does not match...]
    similaridades = cosine_similarity(query_embedding, embeddings_base).flatten()
    
    keyword_score = df.apply(
        lambda x: 1.0 if query.lower() in str(x['pergunta']).lower() or 
                         query.lower() in str(x['resposta']).lower() else 0.0, axis=1
    ).values
    
    df['score'] = (similaridades * 0.7) + (keyword_score * 0.3)
    return df.sort_values(by='score', ascending=False).head(5)

# --- INTERFACE ---
def main():
    st.title("🔍 Central de Ajuda")
    
    modelo = carregar_modelo()
    df = carregar_dados()
    embeddings_base = gerar_embeddings(modelo, df['pergunta'])

    # 2. Categorias na Barra Lateral [1]
    st.sidebar.header("Categorias")
    categorias = ["Todas"] + sorted(df['categoria'].unique().tolist())
    categoria_sel = st.sidebar.radio("Navegue por assunto:", categorias)

    # 3. Campo de Busca com Botão Limpar
    col_busca, col_limpar = st.columns([2, 3])
    
    with col_busca:
        # Usamos session_state para permitir limpar o texto
        if 'input_busca' not in st.session_state:
            st.session_state.input_busca = ""
            
        query = st.text_input("Como podemos ajudar?", value=st.session_state.input_busca, key="campo_busca")

    with col_limpar:
        st.write(" ") # Alinhamento visual
        if st.button("🗑️ Limpar Busca"):
            st.session_state.input_busca = ""
            st.rerun()

    # 4. Processamento de Resultados
    if query:
        resultados = busca_hibrida(query, df, modelo, embeddings_base)
        # Filtro de relevância ajustado para 0.45 conforme conversado
        resultados = resultados[resultados['score'] > 0.45]
        
        if not resultados.empty:
            st.subheader(f"Encontramos {len(resultados)} resultados relevantes:")
            for _, row in resultados.iterrows():
                exibir_faq(row['pergunta'], row['resposta'], expandido=True)
        else:
            st.warning("Nenhum resultado encontrado. Tente outras palavras-chave.")
            
    else:
        # Exibição por Categoria (quando não há busca ativa)
        df_exibicao = df if categoria_sel == "Todas" else df[df['categoria'] == categoria_sel]
        st.subheader(f"Assunto: {categoria_sel}")
        
        for _, row in df_exibicao.iterrows():
            exibir_faq(row['pergunta'], row['resposta'])

# --- FORMATAÇÃO DE TEXTO (Markdown + HTML) ---
def exibir_faq(pergunta, resposta, expandido=False):
    """Aplica formatação: Perguntas negrito, Respostas negrito/itálico, +25% tamanho."""
    # font-size: 1.25rem aumenta a fonte em exatamente 25% [4]
    estilo_p = "font-size: 1.25rem; font-weight: bold;"
    estilo_r = "font-size: 1.25rem; font-weight: bold; font-style: italic;"
    
    with st.expander(pergunta, expanded=expandido):
        st.markdown(f"<p style='{estilo_p}'>{pergunta}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='{estilo_r}'>{resposta}</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()