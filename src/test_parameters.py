# 1. Módulos de sistema e variáveis de ambiente
import os
from dotenv import load_dotenv

# 2. Carregadores e fatiadores do LangChain
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 3. Embeddings, Chroma DB e abstrações de mensagens
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openrouter import ChatOpenRouter

# Carrega a chave de API do arquivo .env
load_dotenv()

print("--- ETAPA 1: Indexando documentos no ChromaDB com Metadados Enriquecidos ---")

# URLs com dados para o experimento
urls = [
    "https://pt.wikipedia.org/wiki/Inteligência_artificial",
    "https://pt.wikipedia.org/wiki/História_da_internet"
]

raw_documents = []
for url in urls:
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        # Adicionando um metadado customizado de categoria/fonte para testar filtros
        for doc in docs:
            doc.metadata["categoria"] = "Wikipedia_PT"
        raw_documents.extend(docs)
    except Exception as e:
        print(f"Erro ao carregar URL {url}: {e}")

# Fatiamento em chunks de 800 caracteres com overlap de 150
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = text_splitter.split_documents(raw_documents)

# Modelo de Embeddings local da HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

# Instancia o Banco Vetorial ChromaDB em memória
vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
print(f"Indexação concluída! Total de chunks: {len(chunks)}")


# ------------------------------------------------------------------
# EXPERIMENTO 1: Variação do Parâmetro top_k (k=1, k=3, k=5)
# ------------------------------------------------------------------
query = "Quais foram os principais marcos da Inteligência Artificial?"
print("\n" + "="*60)
print(f"EXPERIMENTO 1: Testando variação do parâmetro top_k")
print(f"QUERY: '{query}'")
print("="*60)

for k_val in [1, 3, 5]:
    # Cria o retriever com a quantidade k específica
    retriever_k = vector_store.as_retriever(search_kwargs={"k": k_val})
    resultados_k = retriever_k.invoke(query)
    print(f"\n[top_k = {k_val}] Total de trechos retornados: {len(resultados_k)}")
    # Exibe apenas o início do primeiro trecho para comparação
    print(f"  Primeiro trecho: {resultados_k[0].page_content[:150]}...")


# ------------------------------------------------------------------
# EXPERIMENTO 2: Filtro por Limiar de Similaridade (score_threshold)
# ------------------------------------------------------------------
print("\n" + "="*60)
print("EXPERIMENTO 2: Busca por Limiar de Similaridade (score_threshold)")
print("="*60)

# Configura o retriever usando similarity_score_threshold para filtrar conteúdos irrelevantes
try:
    retriever_threshold = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 5}
    )
    resultados_thresh = retriever_threshold.invoke(query)
    print(f"[score_threshold = 0.5] Trechos aprovados no filtro: {len(resultados_thresh)}")
except Exception as e:
    print(f"Aviso de execução no filtro de similaridade: {e}")


# ------------------------------------------------------------------
# EXPERIMENTO 3: Filtragem por Metadados (Metadata Filtering)
# ------------------------------------------------------------------
print("\n" + "="*60)
print("EXPERIMENTO 3: Filtrando por Metadados Especificados")
print("="*60)

# Retriever com filtro exato de chave/valor nos metadados
retriever_meta = vector_store.as_retriever(
    search_kwargs={
        "k": 2,
        "filter": {"categoria": "Wikipedia_PT"}
    }
)
resultados_meta = retriever_meta.invoke(query)
print(f"Trechos recuperados com filtro ['categoria': 'Wikipedia_PT']: {len(resultados_meta)}")
print(f"Metadado do documento retornado: {resultados_meta[0].metadata}")


# ------------------------------------------------------------------
# EXPERIMENTO 4: Prompts Personalizados (Estrito vs Aprofundado)
# ------------------------------------------------------------------
print("\n" + "="*60)
print("EXPERIMENTO 4: Comparando Estilos de Prompts")
print("="*60)

# Recupera o contexto padrão
retriever_padrao = vector_store.as_retriever(search_kwargs={"k": 2})
docs_contexto = retriever_padrao.invoke(query)
contexto_str = "\n\n".join([d.page_content for d in docs_contexto])

# Instancia o LLM com temperatura neutra
llm_padrao = ChatOpenRouter(model="openrouter/free", temperature=0.3)

# Prompt 1: Resposta Estrita e Didática
p1_messages = [
    SystemMessage(content=f"Use APENAS o contexto fornecido para responder de forma simples e didática.\nContexto:\n{contexto_str}"),
    HumanMessage(content=query)
]
resp_p1 = llm_padrao.invoke(p1_messages)

print("\n--- PROMPT 1 (Estrito e Didático) ---")
print(resp_p1.content)

# Prompt 2: Resposta Aprofundada (Combinando Conhecimento Prévio)
p2_messages = [
    SystemMessage(content=f"Combine o contexto fornecido com seu conhecimento prévio e explique os marcos da IA de forma aprofundada.\nContexto:\n{contexto_str}"),
    HumanMessage(content=query)
]
resp_p2 = llm_padrao.invoke(p2_messages)

print("\n--- PROMPT 2 (Combinado e Aprofundado) ---")
print(resp_p2.content)


# ------------------------------------------------------------------
# EXPERIMENTO 5: Personas do Assistente e Ajuste de Temperatura
# ------------------------------------------------------------------
print("\n" + "="*60)
print("EXPERIMENTO 5: Perfis de Assistentes (Temperatura & Estilo)")
print("="*60)

# Perfil 1: Consultor Técnico (Temperatura 0.1 - Baixa variação, foco em exatidão)
consultor_llm = ChatOpenRouter(model="openrouter/free", temperature=0.1)
prompt_consultor = [
    SystemMessage(content=(
        "Você é um Consultor Técnico em IA de perfil extremamente analítico. "
        "Responda de maneira objetiva, estruturada e baseada estritamente no contexto a seguir:\n"
        f"{contexto_str}"
    )),
    HumanMessage(content=query)
]
resp_consultor = consultor_llm.invoke(prompt_consultor)

print("\n [PERFIL 1: Consultor Técnico | Temp=0.1]:")
print(resp_consultor.content)

# Perfil 2: Mentor Criativo (Temperatura 0.8 - Maior diversidade de linguagem e analogias)
mentor_llm = ChatOpenRouter(model="openrouter/free", temperature=0.8)
prompt_mentor = [
    SystemMessage(content=(
        "Você é um Mentor Criativo e entusiasta de tecnologia. "
        "Explique os marcos históricos usando analogias envolventes com base no contexto:\n"
        f"{contexto_str}"
    )),
    HumanMessage(content=query)
]
resp_mentor = mentor_llm.invoke(prompt_mentor)

print("\n [PERFIL 2: Mentor Criativo | Temp=0.8]:")
print(resp_mentor.content)