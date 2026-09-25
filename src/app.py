# ==============================================================================
# IMPORTAÇÃO DAS BIBLIOTECAS
# ==============================================================================
import os                                      # Módulo para verificar arquivos e caminhos no SO
import streamlit as st                         # Framework de interface web
from dotenv import load_dotenv                 # Leitor do arquivo .env

# LangChain - Embeddings, Banco Vetorial e Modelos
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openrouter import ChatOpenRouter

# LangChain - Ferramentas de Busca na Web / Wikipédia
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

# Carrega variáveis de ambiente
load_dotenv()

PERSIST_DIRECTORY = "./chroma_db"
MODEL_EMBEDDING = "mixedbread-ai/mxbai-embed-large-v1"

# ==============================================================================
# CONFIGURAÇÃO DA INTERFACE STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="RAG Híbrido - Local + Web",
    layout="wide"
)

st.title("Chatbot RAG Híbrido: local e web")
st.caption("A aplicação consulta os PDFs locais e pesquisa na Web simultaneamente para fundamentar a resposta.")

# ==============================================================================
# ETAPA 1: INICIALIZAÇÃO DOS RETRIEVERS (LOCAL E WEB)
# ==============================================================================
@st.cache_resource
def carregar_banco_vetorial_local():
    """Carrega o banco de dados vetorial local (ChromaDB) contendo os PDFs da pasta ./data."""
    if not os.path.exists(PERSIST_DIRECTORY) or not os.listdir(PERSIST_DIRECTORY):
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_EMBEDDING,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

@st.cache_resource
def inicializar_ferramenta_web():
    """Inicializa o buscador da Wikipédia para pesquisas em português na internet em tempo real."""
    api_wrapper = WikipediaAPIWrapper(
        lang="pt",                             # Define o idioma para Português
        top_k_results=2,                       # Quantidade de artigos retornados
        doc_content_chars_max=1200             # Limite de caracteres por artigo
    )
    return WikipediaQueryRun(api_wrapper=api_wrapper)

# Instancia as duas fontes de busca
vector_store_local = carregar_banco_vetorial_local()
ferramenta_web = inicializar_ferramenta_web()

# Aviso amigável caso o banco vetorial local ainda não tenha sido gerado
if vector_store_local is None:
    st.sidebar.warning("Banco local não encontrado em `./chroma_db`. Execute `python src/ingest.py` para incluir PDFs locais.")

# ==============================================================================
# ETAPA 2: BARRA LATERAL (CONTROLES E HISTÓRICO)
# ==============================================================================
st.sidebar.header("Fontes de Informação Ativas")

usar_local = st.sidebar.checkbox("Pesquisar na pasta `./data` (PDFs)", value=True)
usar_web = st.sidebar.checkbox("Pesquisar na Internet (Wikipédia)", value=True)

st.sidebar.divider()

top_k_local = st.sidebar.slider(
    "Documentos Locais (top_k):",
    min_value=1,
    max_value=5,
    value=2
)

st.sidebar.divider()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.sidebar.button("🗑️ Limpar Conversa"):
    st.session_state.chat_history = []
    st.rerun()

# ==============================================================================
# ETAPA 3: EXIBIÇÃO DO HISTÓRICO
# ==============================================================================
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# ==============================================================================
# ETAPA 4: PROCESSAMENTO DE PERGUNTAS COMBINANDO AS DUAS FONTES
# ==============================================================================
user_input = st.chat_input("Faça sua pergunta...")

if user_input:
    # Renderiza a pergunta do usuário
    with st.chat_message("user"):
        st.write(user_input)

    contexto_local = ""
    contexto_web = ""
    docs_locais = []

    # --- 1. BUSCA NA BASE LOCAL (PDFs em ./data) ---
    if usar_local and vector_store_local:
        with st.spinner("🔎 Consultando documentos na pasta ./data..."):
            try:
                retriever = vector_store_local.as_retriever(search_kwargs={"k": top_k_local})
                docs_locais = retriever.invoke(user_input)
                if docs_locais:
                    contexto_local = "\n\n".join([f"[PDF Local: {os.path.basename(d.metadata.get('source', 'PDF'))}] {d.page_content}" for d in docs_locais])
            except Exception as e:
                st.warning(f"Aviso na busca local: {e}")

    # --- 2. BUSCA NA INTERNET (WIKIPÉDIA AO VIVO) ---
    if usar_web:
        with st.spinner("Pesquisando na internet ao vivo..."):
            try:
                contexto_web = ferramenta_web.run(user_input)
            except Exception as e:
                contexto_web = "Nenhum resultado encontrado na web."

    # --- 3. PAINEL DE INSPEÇÃO DE FONTES ---
    with st.expander("Fontes Recuperadas (Inspeção Transparente)"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Fontes da Pasta `data/`")
            if docs_locais:
                for idx, doc in enumerate(docs_locais, 1):
                    arquivo = os.path.basename(doc.metadata.get('source', 'PDF'))
                    pag = doc.metadata.get('page', 'N/A')
                    st.markdown(f"**Trecho {idx}:** `{arquivo}` (Pág. {pag})")
                    st.caption(doc.page_content[:300] + "...")
            else:
                st.info("Nenhum trecho local recuperado ou busca local desativada.")

        with col2:
            st.subheader("Fontes da Internet")
            if contexto_web and "No good Wikipedia Search Result" not in contexto_web:
                st.caption(contexto_web[:600] + "...")
            else:
                st.info("Nenhum trecho web relevante encontrado para esta consulta.")

    # --- 4. FUSÃO DO CONTEXTO E PROMPT FINAL ---
    contexto_combinado = f"""
=== FONTE 1: DOCUMENTOS LOCAIS (PASTA DATA) ===
{contexto_local if contexto_local else 'Nenhum documento local encontrado.'}

=== FONTE 2: PESQUISA DA INTERNET (WEB) ===
{contexto_web if contexto_web and 'No good' not in contexto_web else 'Nenhum resultado relevante na web.'}
"""

    system_prompt = SystemMessage(content=(
        "Você é um assistente RAG Híbrido avançado. Responda à pergunta do usuário utilizando "
        "as informações fornecidas nas duas fontes acima (Documentos Locais e Internet).\n"
        "Sempre mencione na sua resposta de onde veio a informação (ex: 'Segundo os documentos locais...' ou 'De acordo com informações da web...').\n"
        "Se nenhuma das duas fontes contiver a resposta, declare explicitamente que a informação não foi localizada."
        f"\n\nCONTEXTO DISPONÍVEL:\n{contexto_combinado}"
    ))

    # --- 5. GERAÇÃO DA RESPOSTA VIA OPENROUTER ---
    with st.chat_message("assistant"):
        with st.spinner("Sintetizando resposta unificada..."):
            try:
                llm = ChatOpenRouter(
                    model="openrouter/free",
                    temperature=0.2
                )
                resposta = llm.invoke([system_prompt, HumanMessage(content=user_input)])
                st.write(resposta.content)

                # Armazena no histórico de memória
                st.session_state.chat_history.append(HumanMessage(content=user_input))
                st.session_state.chat_history.append(AIMessage(content=resposta.content))

            except Exception as e:
                st.error(f"Erro de comunicação com a API: {e}")