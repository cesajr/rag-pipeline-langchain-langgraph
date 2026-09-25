import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List

# 1. Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Configura o ChatOpenAI para apontar para o OpenRouter
# Dica: Você pode usar modelos gratuitos ou de alta performance do OpenRouter (ex: "google/gemini-flash-1.5")
llm = ChatOpenAI(
base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="google/gemini-2.5-flash",  # Ou outro modelo de sua preferência
    temperature=0.0,
    max_tokens=1000  # Limita explicitamente a resposta para evitar o erro de cota/tokens
)

print("--- ETAPA 1: Carregando artigos da Wikipedia ---")
# 2. Escolha dois temas e carregue com WikipediaLoader (definimos lang='pt' para buscar em português)
topics = ["Inteligência Artificial", "História da Internet"]
raw_documents = []

for topic in topics:
    print(f"Carregando artigo: {topic}...")
    loader = WikipediaLoader(query=topic, load_max_docs=1, lang="pt")
    docs = loader.load()
    raw_documents.extend(docs)

print(f"Total de documentos carregados: {len(raw_documents)}")

print("\n--- ETAPA 2: Dividindo os textos em chunks ---")
# 3. Divisão dos textos em pedaços menores (chunks)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(raw_documents)
print(f"Total de chunks gerados: {len(chunks)}")

print("\n--- ETAPA 3: Criando embeddings e Banco Vetorial Chroma ---")
# 4 & 5. Gerar embeddings locais com HuggingFace e salvar no ChromaDB
embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

# Criação do banco vetorial em memória/local
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="mini_rag_wikipedia"
)

# Cria o retriever para buscar os trechos mais relevantes
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

print("\n--- ETAPA 4: Definindo o Estado e o Grafo com LangGraph ---")

# Define a estrutura de dados (estado) que flui pelo grafo
class AgentState(TypedDict):
    pergunta: str
    contexto: str
    resposta: str

# Nó 1: Busca no banco vetorial
def buscar_contexto(state: AgentState):
    pergunta = state["pergunta"]
    print(f"\n[Nó: Busca] Pesquisando trechos para: '{pergunta}'")
    docs_relacionados = retriever.invoke(pergunta)
    
    # Junta o conteúdo dos chunks encontrados em uma única string de contexto
    contexto_unido = "\n\n".join([doc.page_content for doc in docs_relacionados])
    return {"contexto": contexto_unido}

# Nó 2 & 3: Inclusão de contexto no prompt e geração da resposta com o LLM via OpenRouter
def gerar_resposta(state: AgentState):
    pergunta = state["pergunta"]
    contexto = state["contexto"]
    print("[Nó: Geração] Enviando contexto e pergunta para o LLM...")
    
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
    
    response = llm.invoke(messages)
    return {"resposta": response.content}

# Construindo o Grafo
workflow = StateGraph(AgentState)

# Adicionando os nós
workflow.add_node("buscar_contexto", buscar_contexto)
workflow.add_node("gerar_resposta", gerar_resposta)

# Definindo as arestas (fluxo sequencial)
workflow.add_edge(START, "buscar_contexto")
workflow.add_edge("buscar_contexto", "gerar_resposta")
workflow.add_edge("gerar_resposta", END)

# Compilando o grafo
app_rag = workflow.compile()

print("\n--- ETAPA 5: Modo Interativo de Consultas ---")
print("Digite sua pergunta (ou digite 'sair' para encerrar):")

while True:
    pergunta_usuario = input("\nPergunta: ")
    if pergunta_usuario.lower() == 'sair':
        print("Encerrando o mini-RAG. Até logo!")
        break
    
    if not pergunta_usuario.strip():
        continue

    # Executa o grafo com a pergunta inserida pelo usuário
    resultado = app_rag.invoke({"pergunta": pergunta_usuario})
    
    print("-" * 50)
    print(f"RESPOSTA:\n{resultado['resposta']}")
    print("-" * 50)