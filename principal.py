import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import unicodedata
import difflib

# 1. Configuração da Página
st.set_page_config(layout="wide", page_title="FAQ Inteligente", page_icon="🔍")

# --- FUNÇÕES DE SUPORTE ---

def normalizar_texto(texto):
    """Padroniza o texto removendo acentos para busca flexível."""
    if not isinstance(texto, str): return ""
    texto = texto.lower().strip()
    texto = unicodedata.normalize('NFD', texto)
    return "".join([c for c in texto if unicodedata.category(c) != 'Mn'])

@st.cache_resource
def carregar_modelo():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def carregar_dados():
    df = pd.read_excel("faq.xlsx")
    df.columns = df.columns.str.strip()
    return df.dropna(subset=['pergunta', 'resposta'])

@st.cache_data
def gerar_base_embeddings(_modelo, perguntas):
    return _modelo.encode(perguntas.tolist())

# --- LÓGICA DE INTERAÇÃO ---

def limpar_busca():
    st.session_state.query_input = ""

def busca_hibrida(query, df, modelo, embeddings_base):
    query_norm = normalizar_texto(query)
    query_embedding = modelo.encode([query])
    # .flatten() evita erro de dimensão entre matriz e array [Turno 7]
    similaridades = cosine_similarity(query_embedding, embeddings_base).flatten()
    
    keyword_score = df.apply(
        lambda x: 1.0 if query_norm in normalizar_texto(str(x['pergunta'])) or 
                         query_norm in normalizar_texto(str(x['resposta'])) else 0.0, axis=1
    ).values
    
    df['score'] = (similaridades * 0.7) + (keyword_score * 0.3)
    return df.sort_values(by='score', ascending=False).head(5)

# --- COMPONENTE DE EXIBIÇÃO ---

def exibir_faq(pergunta, resposta, expandido=False):
    """Exibe a FAQ com resposta em 18px, negrito, itálico e fundo colorido."""
    estilo_bloco_resposta = """
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #007bff;
        font-size: 18px;
        font-weight: bold;
        font-style: italic;
        color: #1e1e1e;
        line-height: 1.6;
    """
    with st.expander(f"**{pergunta}**", expanded=expandido):
        st.markdown(f"<div style='{estilo_bloco_resposta}'>{resposta}</div>", unsafe_allow_html=True)

# --- INTERFACE PRINCIPAL ---

def main():
    st.title("🔍 Central de Ajuda")

    modelo = carregar_modelo()
    df = carregar_dados()
    embeddings_base = gerar_base_embeddings(modelo, df['pergunta'])

    if 'query_input' not in st.session_state:
        st.session_state.query_input = ""

    # 2. Barra Lateral: Categorias
    st.sidebar.image("imagens/logo_marca_caj.png", use_container_width=True)
    st.sidebar.header("Categorias")
    categorias = ["Todas"] + sorted(df['categoria'].unique().tolist())
    categoria_sel = st.sidebar.radio("Navegue por assunto:", categorias)

    # 3. Busca Ampliada (Proporção 4:1) [1]
    col_busca, col_limpar = st.columns([2, 3]) 
    
    with col_busca:
        query = st.text_input(
            "Como podemos ajudar?", 
            key="query_input", 
            placeholder="Digite sua dúvida..."
        )

    with col_limpar:
        st.write(" ") # Alinhamento visual
        st.button("🗑️ Limpar", on_click=limpar_busca, use_container_width=True)

    # 4. Processamento de Resultados
    if query:
        resultados = busca_hibrida(query, df, modelo, embeddings_base)
        resultados_filtrados = resultados[resultados['score'] > 0.35]
        
        if not resultados_filtrados.empty:
            st.subheader(f"Resultados Relevantes ({len(resultados_filtrados)}):")
            for _, row in resultados_filtrados.iterrows():
                exibir_faq(row['pergunta'], row['resposta'], expandido=True)
        else:
            sugestoes = difflib.get_close_matches(query, df['pergunta'].tolist(), n=1, cutoff=0.5)
            if sugestoes:
                if st.button(f"🔍 Você quis dizer: '{sugestoes}'?"):
                    st.session_state.query_input = sugestoes
                    st.rerun()
            else:
                st.warning("Nenhum resultado encontrado.")
    else:
        st.subheader(f"Assunto: {categoria_sel}")
        df_exibicao = df if categoria_sel == "Todas" else df[df['categoria'] == categoria_sel]
        for _, row in df_exibicao.iterrows():
            exibir_faq(row['pergunta'], row['resposta'])

    st.sidebar.markdown("---")
    st.sidebar.caption("⭐ Desenvolvido por Luciano Reis")

if __name__ == "__main__":
    main()