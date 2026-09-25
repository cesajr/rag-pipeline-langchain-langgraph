import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter  # Integração oficial do OpenRouter
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 1. Carrega as variáveis de ambiente do arquivo .env (onde fica sua OPENROUTER_API_KEY)
load_dotenv()

# Inicializa o modelo de linguagem utilizando o ChatOpenRouter nativo
# O model="openrouter/free" usa o roteador inteligente para escolher um modelo gratuito ativo
modelo = ChatOpenRouter(
    model="openrouter/free",
    temperature=0.3
)

print("--- ETAPA 1: Carregando artigos da Wikipedia ---")
# 2. Escolha dos temas e carregamento via WikipediaLoader em português
topics = ["Inteligência Artificial", "História da Internet"]
raw_documents = []

for topic in topics:
    print(f"Carregando artigo: {topic}...")
    loader = WikipediaLoader(query=topic, load_max_docs=1, lang="pt")
    docs = loader.load()
    raw_documents.extend(docs)

print(f"Total de documentos carregados: {len(raw_documents)}")

print("\n--- ETAPA 2: Dividindo os textos em chunks ---")
# 3. Divisão dos textos em pedaços menores (chunks) para otimizar as buscas semânticas
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(raw_documents)
print(f"Total de chunks gerados: {len(chunks)}")

print("\n--- ETAPA 3: Criando embeddings e Banco Vetorial Chroma ---")
# 4 & 5. Gerar embeddings locais com HuggingFace e salvar no ChromaDB
embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="mini_rag_wikipedia"
)

# Cria o retriever para buscar os trechos mais relevantes (k=3)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

print("\n--- ETAPA 4: Definindo o Estado e o Grafo com LangGraph ---")

# Define a estrutura de dados (estado) que flui pelas etapas do grafo
class AgentState(TypedDict):
    pergunta: str
    contexto: str
    resposta: str

# Nó 1: Busca no banco vetorial com base na pergunta
def buscar_contexto(state: AgentState):
    pergunta = state["pergunta"]
    print(f"\n[Nó: Busca] Pesquisando trechos para: '{pergunta}'")
    docs_relacionados = retriever.invoke(pergunta)
    
    # Junta o conteúdo dos chunks encontrados em uma única string de contexto
    contexto_unido = "\n\n".join([doc.page_content for doc in docs_relacionados])
    return {"contexto": contexto_unido}

# Nó 2: Geração de resposta com o LLM via ChatOpenRouter
def gerar_resposta(state: AgentState):
    pergunta = state["pergunta"]
    contexto = state["contexto"]
    print("[Nó: Geração] Enviando contexto e pergunta para o modelo via OpenRouter...")
    
    system_prompt = (
        "Você é um assistente especialista. Responda à pergunta do usuário baseando-se "
        "estritamente nas informações fornecidas no contexto abaixo. Se a resposta não estiver "
        "no contexto, diga que não encontrou a informação.\n\n"
        f"Contexto:\n{contexto}"
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=pergunta)
    ]
    
    # Chamada ao modelo OpenRouter configurado
    response = modelo.invoke(messages)
    return {"resposta": response.content}

# Construindo o Grafo de Execução com LangGraph
workflow = StateGraph(AgentState)

# Adicionando os nós ao grafo
workflow.add_node("buscar_contexto", buscar_contexto)
workflow.add_node("gerar_resposta", gerar_resposta)

# Definindo as arestas (o fluxo sequencial de execução)
workflow.add_edge(START, "buscar_contexto")
workflow.add_edge("buscar_contexto", "gerar_resposta")
workflow.add_edge("gerar_resposta", END)

# Compilando o grafo para torná-lo executável
app_rag = workflow.compile()

print("\n--- ETAPA 5: Executando uma Pergunta de Teste ---")
pergunta_teste = "Quando surgiu o termo Inteligência Artificial?"

# Executando o fluxo completo do RAG
resultado = app_rag.invoke({"pergunta": pergunta_teste})

print("\n" + "="*50)
print(f"PERGUNTA: {pergunta_teste}")
print("="*50)
print(f"RESPOSTA GERADA:\n{resultado['resposta']}")
print("="*50)