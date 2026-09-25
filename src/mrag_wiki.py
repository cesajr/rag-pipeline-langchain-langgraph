import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WikipediaLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_community.document_loaders import WebBaseLoader

# 1. Carrega as variáveis de ambiente (.env)
load_dotenv()

# Inicializa o modelo via ChatOpenRouter
modelo = ChatOpenRouter(
    model="openrouter/free",
    temperature=0.3
)

print("--- ETAPA 1: Carregando fontes de dados da Web ---")
raw_documents = []

# URLs oficiais dos artigos da Wikipedia em português
urls = [
    "https://pt.wikipedia.org/wiki/Inteligência_artificial",
    "https://pt.wikipedia.org/wiki/História_da_internet"
]

for url in urls:
    try:
        print(f"Carregando página: {url}...")
        loader = WebBaseLoader(url)
        raw_documents.extend(loader.load())
    except Exception as e:
        print(f"Erro ao carregar a URL {url}: {e}")

print(f"Total de documentos carregados: {len(raw_documents)}")

# Fonte A: Artigos da Wikipedia com tratamento de erro
#topics = ["Artificial intelligence", "History of the internet"]  # Usar em inglês costuma ser mais estável na API

#for topic in topics:
#    try:
#        print(f"Carregando Wikipedia: {topic}...")
#        # Definimos lang="en" para maior estabilidade, ou mantemos "pt"
#        loader_wiki = WikipediaLoader(query=topic, load_max_docs=1, lang="en")
#        raw_documents.extend(loader_wiki.load())
#    except Exception as e:
#        print(f"Aviso: Não foi possível carregar o tópico '{topic}' da Wikipedia ({e}).")

# Se por acaso a Wikipedia falhar completamente, adicionamos um texto de fallback para não quebrar o teste
#if not raw_documents:
#    print("Usando documento de contingência local...")
#    from langchain_core.documents import Document
#    raw_documents = [
#        Document(page_content="A Inteligência Artificial foi fundada como disciplina acadêmica em 1956 na Dartmouth Conference.")
#    ]

#print(f"Total de documentos carregados: {len(raw_documents)}")

# Fonte B: Exemplo de inclusão de arquivos locais (Opcional)
# Se você tiver um arquivo de texto ou PDF na mesma pasta, basta descomentar as linhas abaixo:
# try:
#     loader_txt = TextLoader("meu_documento.txt", encoding="utf-8")
#     raw_documents.extend(loader_txt.load())
#     print("Arquivo local .txt carregado com sucesso!")
# except Exception:
#     pass # Ignora se o arquivo não existir

print(f"Total de documentos carregados de todas as fontes: {len(raw_documents)}")

print("\n--- ETAPA 2: Dividindo os textos em chunks ---")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(raw_documents)
print(f"Total de chunks gerados: {len(chunks)}")

print("\n--- ETAPA 3: Criando embeddings e Banco Vetorial Chroma ---")
embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="mini_rag_com_memoria"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

print("\n--- ETAPA 4: Definindo o Estado, Grafo e Memória ---")

# Estado estruturado para acumular o histórico de mensagens automaticamente ('add_messages')
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    contexto: str

# Nó 1: Busca o contexto com base na última mensagem do usuário
def buscar_contexto(state: AgentState):
    ultima_mensagem = state["messages"][-1].content
    print(f"\n[Nó: Busca] Pesquisando trechos para: '{ultima_mensagem}'")
    
    docs_relacionados = retriever.invoke(ultima_mensagem)
    contexto_unido = "\n\n".join([doc.page_content for doc in docs_relacionados])
    return {"contexto": contexto_unido}

# Nó 2: Gera a resposta considerando o contexto e o histórico de conversação
def gerar_resposta(state: AgentState):
    historico = state["messages"]
    contexto = state["contexto"]
    print("[Nó: Geração] Enviando histórico e contexto para o modelo...")
    
    system_prompt = SystemMessage(content=(
        "Você é um assistente especialista em RAG. Responda à pergunta do usuário baseando-se "
        "estritamente nas informações fornecidas no contexto abaixo e no histórico da conversa.\n\n"
        f"Contexto recuperado:\n{contexto}"
    ))
    
    # Combina as instruções do sistema com toda a conversa anterior
    mensagens_para_llm = [system_prompt] + historico
    response = modelo.invoke(mensagens_para_llm)
    
    # Retorna a nova resposta da IA para ser adicionada ao histórico
    return {"messages": [response]}

# Construindo o Grafo
workflow = StateGraph(AgentState)
workflow.add_node("buscar_contexto", buscar_contexto)
workflow.add_node("gerar_resposta", gerar_resposta)

workflow.add_edge(START, "buscar_contexto")
workflow.add_edge("buscar_contexto", "gerar_resposta")
workflow.add_edge("gerar_resposta", END)

# Compilando o grafo com o MemorySaver para reter o histórico por thread
memory = MemorySaver()
app_rag = workflow.compile(checkpointer=memory)

print("\n--- ETAPA 5: Testando a Conversa com Memória ---")

# Definimos um identificador de sessão (thread)
config = {"configurable": {"thread_id": "sessao_estudo_01"}}

# Pergunta 1
perq_1 = "Quando surgiu o termo Inteligência Artificial?"
print(f"\nUsuário: {perq_1}")
res_1 = app_rag.invoke({"messages": [HumanMessage(content=perq_1)]}, config)
print(f"IA: {res_1['messages'][-1].content}")

# Pergunta 2 (Fazendo uma referência implícita que exige memória da Pergunta 1)
perq_2 = "Quais foram os principais nomes ou eventos associados a esse marco?"
print(f"\nUsuário: {perq_2}")
res_2 = app_rag.invoke({"messages": [HumanMessage(content=perq_2)]}, config)
print(f"IA: {res_2['messages'][-1].content}")