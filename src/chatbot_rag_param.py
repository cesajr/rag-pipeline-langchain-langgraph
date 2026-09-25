# 1. Módulos do sistema e interface web
import os
import streamlit as st
from dotenv import load_dotenv

# 2. Carregadores, fatiadores e vetorização do LangChain
from langchain_community.document_loaders import WebBaseLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 3. Mensagens e conexão com o OpenRouter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter

# Carrega as variáveis de ambiente (.env)
load_dotenv()

# Caminho para salvar/carregar o banco vetorial persistentemente no disco
PERSIST_DIRECTORY = "./chroma_db"

# Configuração visual da página
st.set_page_config(page_title="RAG - Ajuste de Parâmetros", page_icon="⚙️", layout="wide")

st.title("⚙️ Chatbot RAG: Ajuste Dinâmico de Parâmetros de Busca")
st.caption("Experimente como a variação de top_k, score_threshold e metadados impacta a recuperação de dados e a resposta da IA.")

# ------------------------------------------------------------------
# ETAPA 1: Carregamento/Persistência do Banco Vetorial (ChromaDB)
# ------------------------------------------------------------------
@st.cache_resource
def carregar_banco_vetorial():
    """Carrega o ChromaDB do disco se já existir. Caso contrário, baixa os dados e salva na pasta ./chroma_db."""
    embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

    # Se a pasta do ChromaDB já existe e contém arquivos, carrega diretamente do disco
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        st.sidebar.success("📦 Banco vetorial carregado do disco (`./chroma_db`)")
        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )

    # Caso o banco não exista no disco, faz a ingestão inicial
    st.sidebar.info("🌐 Criando novo banco vetorial e salvando em disco...")
    raw_documents = []

    # 1. Se houver PDFs na pasta ./data, prioriza os PDFs locais
    if os.path.exists("./data") and any(f.endswith(".pdf") for f in os.listdir("./data")):
        try:
            loader = PyPDFDirectoryLoader("./data")
            pdf_docs = loader.load()
            for doc in pdf_docs:
                doc.metadata["fonte"] = "PDF Local"
            raw_documents.extend(pdf_docs)
        except Exception as e:
            st.error(f"Erro ao carregar PDFs da pasta ./data: {e}")

    # 2. Adiciona dados da Wikipedia com User-Agent para evitar bloqueio 403
    urls = [
        "https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial",
        "https://pt.wikipedia.org/wiki/Hist%C3%B3ria_da_internet"
    ]
    for url in urls:
        try:
            loader = WebBaseLoader(
                web_paths=(url,),
                header_template={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"
                }
            )
            docs = loader.load()
            for doc in docs:
                doc.metadata["fonte"] = "Wikipedia"
            raw_documents.extend(docs)
        except Exception as e:
            st.error(f"Erro ao carregar a URL {url}: {e}")

    if not raw_documents:
        st.error("Nenhum documento pôde ser carregado.")
        return None

    # Fatiamento do texto em chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(raw_documents)

    # Cria e salva persistentemente no diretório ./chroma_db
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    return vectorstore

with st.spinner("Inicializando banco vetorial (ChromaDB)..."):
    vector_store = carregar_banco_vetorial()


# ------------------------------------------------------------------
# ETAPA 2: Barra Lateral - Controles Dinâmicos dos Parâmetros
# ------------------------------------------------------------------
st.sidebar.header("🎛️ Parâmetros do Retriever")

top_k = st.sidebar.slider(
    "Número de documentos (top_k):",
    min_value=1,
    max_value=10,
    value=3,
    help="Quantidade máxima de trechos recuperados do banco vetorial."
)

score_threshold = st.sidebar.slider(
    "Limite mínimo de similaridade (score_threshold):",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.05,
    help="Descarta trechos com grau de similaridade semântica inferior ao limite."
)

fonte_filtro = st.sidebar.text_input(
    "Filtrar por fonte (metadados):",
    placeholder="Ex: Wikipedia ou PDF Local",
    help="Aplica um filtro exato no metadado 'fonte'."
)

st.sidebar.divider()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.sidebar.button("🗑️ Limpar Histórico de Conversa"):
    st.session_state.chat_history = []
    st.rerun()


# ------------------------------------------------------------------
# ETAPA 3: Configuração do Retriever
# ------------------------------------------------------------------
if vector_store is not None:
    search_kwargs = {"k": top_k}

    if fonte_filtro.strip():
        search_kwargs["filter"] = {"fonte": fonte_filtro.strip()}

    if score_threshold > 0.0:
        search_kwargs["score_threshold"] = score_threshold
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=search_kwargs
        )
    else:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
else:
    retriever = None


# ------------------------------------------------------------------
# ETAPA 4: Interface do Chat e Processamento da Resposta
# ------------------------------------------------------------------
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

user_input = st.chat_input("Faça sua pergunta sobre IA, Internet ou PDFs da base...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    # 1. Recuperação no ChromaDB
    docs_recuperados = []
    if retriever:
        try:
            docs_recuperados = retriever.invoke(user_input)
        except Exception as e:
            st.error(f"Erro na busca vetorial: {e}")

    # Exibe os trechos recuperados na tela
    with st.expander(f"🔍 Chunks Recuperados pelo Retriever ({len(docs_recuperados)} trechos encontrados)"):
        if not docs_recuperados:
            st.warning("Nenhum trecho atendeu aos critérios. Dicas:\n- Reduza o 'score_threshold'\n- Deixe a 'fonte' em branco se não tiver certeza do nome exato.")
        else:
            for i, doc in enumerate(docs_recuperados, 1):
                st.markdown(f"**Trecho {i}** | Fonte: `{doc.metadata.get('fonte', 'N/A')}`")
                st.caption(doc.page_content)
                st.divider()

    # 2. Resposta do LLM
    contexto_texto = "\n\n".join([d.page_content for d in docs_recuperados])
    
    system_prompt = SystemMessage(content=(
        "Você é um assistente RAG preciso. Responda à pergunta do usuário estritamente com base "
        "no contexto fornecido abaixo. Se o contexto estiver vazio ou não contiver a resposta, "
        "informe explicitamente que não encontrou informações nos documentos recuperados.\n\n"
        f"CONTEXTO RECUPERADO:\n{contexto_texto if contexto_texto else 'Nenhum documento atendeu aos critérios de busca.'}"
    ))

    with st.chat_message("assistant"):
        with st.spinner("Gerando resposta..."):
            llm = ChatOpenRouter(
                model="openrouter/free",
                temperature=0.2
            )
            
            resposta = llm.invoke([system_prompt, HumanMessage(content=user_input)])
            st.write(resposta.content)

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=resposta.content))