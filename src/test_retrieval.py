# 1. Módulos para manipulação do sistema e variáveis de ambiente
import os
from dotenv import load_dotenv

# 2. Módulos do LangChain para carregamento e divisão de dados não estruturados
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 3. Módulos do LangChain para embeddings e banco de dados vetorial local
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 4. Módulos para estruturação das mensagens e chamada do modelo de linguagem (LLM)
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openrouter import ChatOpenRouter

# 5. Módulos do LangGraph e typing para construção do fluxo orquestrado via grafos
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# Carrega as variáveis declaradas no arquivo .env para o ambiente de execução do Python
load_dotenv()

# Inicializa o modelo de linguagem (LLM) utilizando a integração do OpenRouter com temperatura baixa (0.2) para respostas mais objetivas
modelo = ChatOpenRouter(
    model="openrouter/free",
    temperature=0.2
)

print("--- ETAPA 1: Preparando o Banco Vetorial ---")

# Lista com as URLs oficiais das páginas que servirão de fonte de dados
urls = [
    "https://pt.wikipedia.org/wiki/Inteligência_artificial",
    "https://pt.wikipedia.org/wiki/História_da_internet"
]

# Lista onde serão guardados os documentos brutos carregados
raw_documents = []

# Loop para iterar e carregar o conteúdo HTML de cada URL
for url in urls:
    try:
        # Instancia o carregador para a URL específica
        loader = WebBaseLoader(url)
        # Executa o carregamento e adiciona os documentos extraídos à lista principal
        raw_documents.extend(loader.load())
    except Exception as e:
        # Captura e exibe falhas na raspagem da página sem interromper o script
        print(f"Aviso ao carregar {url}: {e}")

# Instancia o fatiador de texto: pedaços de 800 caracteres com sobreposição (overlap) de 150 caracteres
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

# Realiza a divisão dos documentos brutos em chunks menores
chunks = text_splitter.split_documents(raw_documents)

# Carrega o modelo local da HuggingFace responsável pela conversão dos chunks de texto em vetores numéricos (embeddings)
embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

# Cria e popula o banco vetorial Chroma na memória utilizando os chunks e o modelo de embeddings criado
vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)

print(f"Banco vetorial pronto com {len(chunks)} chunks!")

# ------------------------------------------------------------------
# EXPERIMENTO 1: Comparando Resposta SEM Contexto vs COM Contexto
# ------------------------------------------------------------------

# Pergunta de teste que será submetida aos dois cenários
pergunta = "Quais foram as contribuições de John McCarthy para a área de tecnologia?"

print("\n" + "="*60)
print(f"PERGUNTA DE TESTE: {pergunta}")
print("="*60)

# 1. Execução do modelo enviando apenas a pergunta direta (sem consultar o banco vetorial)
print("\n[1] GERANDO RESPOSTA SEM CONTEXTO (Apenas conhecimento geral do LLM):")
resposta_direta = modelo.invoke([HumanMessage(content=pergunta)])
# Exibe os primeiros 400 caracteres da resposta gerada estritamente pelo conhecimento interno do LLM
print(resposta_direta.content[:400] + "...\n")

# 2. Configuração da busca semântica: transforma o banco vetorial em um componente de recuperação (retriever) para trazer 3 trechos
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Executa a busca vetorial usando a pergunta do usuário
docs_recuperados = retriever.invoke(pergunta)

print("\n[2] BUSCA VETORIAL - TRECHOS RECUPERADOS (k=3):")
# Itera sobre os documentos retornados para exibir o trecho e a origem da informação (metadados)
for i, doc in enumerate(docs_recuperados, 1):
    fonte = doc.metadata.get("source", "Desconhecida")
    print(f"\n--- Trecho {i} [Fonte: {fonte}] ---")
    print(doc.page_content[:200] + "...")

# Consolida os trechos retornados do banco vetorial em uma única string formatada com citação da fonte
contexto_texto = "\n\n".join([f"- {d.page_content} (Fonte: {d.metadata.get('source')})" for d in docs_recuperados])

# Cria o prompt estruturado combinando instruções estritas de limitação de conhecimento ao contexto e citação de fontes
prompt_com_contexto = [
    SystemMessage(content=(
        "Você é um assistente preciso. Responda à pergunta do usuário APENAS com base "
        "nas informações fornecidas abaixo. Ao final, cite explicitamente as fontes utilizadas.\n\n"
        f"CONTEXTO DISPONÍVEL:\n{contexto_texto}"
    )),
    HumanMessage(content=pergunta)
]

# Envia o prompt com o contexto injetado para o modelo de linguagem
print("\n[3] GERANDO RESPOSTA COM CONTEXTO E CITAÇÃO DE FONTES:")
resposta_rag = modelo.invoke(prompt_com_contexto)
# Exibe a resposta final fundamentada no contexto do banco vetorial
print(resposta_rag.content)


# ------------------------------------------------------------------
# EXPERIMENTO 2: Testando via LangGraph + Variação de K
# ------------------------------------------------------------------
print("\n" + "="*60)
print("EXPERIMENTO 2: Fluxo Orquestrado via LangGraph (Avaliando k=6)")
print("="*60)

# Define a estrutura de dados fortemente tipada que trafegará através dos nós do grafo
class RAGState(TypedDict):
    pergunta: str
    k_valor: int
    contexto: str
    resposta: str

# Função que define o Nó de Busca no grafo
def no_busca(state: RAGState):
    # Obtém o valor de 'k' (quantidade de documentos) do estado, assumindo 3 como padrão caso não esteja definido
    k = state.get("k_valor", 3)
    # Cria dinamicamente um retriever configurado com a quantidade 'k' de trechos solicitados
    retriever_dinamico = vector_store.as_retriever(search_kwargs={"k": k})
    # Executa a busca no ChromaDB
    docs = retriever_dinamico.invoke(state["pergunta"])
    
    print(f"\n[Nó Busca] Recuperados {len(docs)} chunks (k={k}).")
    # Une o conteúdo dos documentos no campo 'contexto' do estado
    contexto = "\n\n".join([f"[{d.metadata.get('source')}] {d.page_content}" for d in docs])
    # Retorna a atualização da chave 'contexto' no estado do grafo
    return {"contexto": contexto}

# Função que define o Nó de Geração da resposta no grafo
def no_geracao(state: RAGState):
    # Monta a estrutura de mensagens utilizando o contexto que foi alimentado pelo nó de busca anterior
    prompt = [
        SystemMessage(content=(
            "Responda estritamente com base no contexto fornecido e cite a fonte no final.\n\n"
            f"Contexto:\n{state['contexto']}"
        )),
        HumanMessage(content=state["pergunta"])
    ]
    # Executa o modelo de linguagem com o prompt contendo o contexto recuperado
    resp = modelo.invoke(prompt)
    # Retorna a atualização da chave 'resposta' no estado do grafo
    return {"resposta": resp.content}

# Cria o construtor do grafo utilizando a classe de estado definida (RAGState)
graph_builder = StateGraph(RAGState)

# Adiciona os dois nós sequenciais ao fluxo do grafo
graph_builder.add_node("buscar", no_busca)
graph_builder.add_node("gerar", no_geracao)

# Conecta as arestas para definir a ordem de execução: Início -> Busca -> Geração -> Fim
graph_builder.add_edge(START, "buscar")
graph_builder.add_edge("buscar", "gerar")
graph_builder.add_edge("gerar", END)

# Compila o grafo para deixá-lo pronto para invocação
app = graph_builder.compile()

# Executa o fluxo do grafo passando a pergunta e o valor k=6 para avaliar o impacto da injeção de mais chunks de contexto
resultado_k6 = app.invoke({"pergunta": pergunta, "k_valor": 6})

# Exibe o resultado final processado pela pipeline do LangGraph
print("\n[Resposta do Grafo com k=6]:")
print(resultado_k6["resposta"])