# src/ingest.py
import os
import time
from dotenv import load_dotenv

# Carregadores do LangChain
from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_DIRECTORY = "./data"
PERSIST_DIRECTORY = "./chroma_db"
MODEL_EMBEDDING = "mixedbread-ai/mxbai-embed-large-v1"

# Lista de URLs da Wikipedia para ingestão automática
URLS_WIKIPEDIA = [
    "https://pt.wikipedia.org/wiki/Aprendizagem_por_refor%C3%A7o",
    "https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial",
    "https://pt.wikipedia.org/wiki/Hist%C3%B3ria_da_internet"
]

def executar_ingestao_mista():
    tempo_inicio = time.time()
    print("Iniciando ingestão combinada (PDFs Locais + Wikipedia)...\n")
    
    documentos_totais = []

    # --------------------------------------------------------------------------
    # 1. CARREGAMENTO DOS PDFs LOCAIS
    # --------------------------------------------------------------------------
    if os.path.exists(DATA_DIRECTORY):
        arquivos_pdf = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith(".pdf")]
        if arquivos_pdf:
            print(f"Carregando {len(arquivos_pdf)} arquivo(s) PDF de '{DATA_DIRECTORY}'...")
            try:
                loader_pdf = PyPDFDirectoryLoader(DATA_DIRECTORY)
                docs_pdf = loader_pdf.load()
                # Adiciona metadado para diferenciar a origem no Retriever
                for doc in docs_pdf:
                    doc.metadata["fonte"] = "PDF Local"
                documentos_totais.extend(docs_pdf)
                print(f"   └─ {len(docs_pdf)} páginas de PDF extraídas.")
            except Exception as e:
                print(f"Erro ao carregar PDFs locais: {e}")
        else:
            print(f"Nenhum PDF encontrado em '{DATA_DIRECTORY}'. Processando apenas URLs.")
    else:
        os.makedirs(DATA_DIRECTORY)
        print(f"Pasta '{DATA_DIRECTORY}' criada.")

    # --------------------------------------------------------------------------
    # 2. CARREGAMENTO DAS PÁGINAS WEB (WIKIPEDIA)
    # --------------------------------------------------------------------------
    print(f"Carregando {len(URLS_WIKIPEDIA)} página(s) da Web...")
    for url in URLS_WIKIPEDIA:
        try:
            # Header simulando navegador para evitar bloqueio 403 HTTP da Wikipedia
            loader_web = WebBaseLoader(
                web_paths=(url,),
                header_template={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"
                }
            )
            docs_web = loader_web.load()
            for doc in docs_web:
                doc.metadata["fonte"] = "Wikipedia"
            documentos_totais.extend(docs_web)
            print(f"   └─ Sucesso: {url}")
        except Exception as e:
            print(f"⚠️ Falha ao carregar a URL '{url}': {e}")

    if not documentos_totais:
        print("Nenhum documento foi carregado. Abortando ingestão.")
        return

    # --------------------------------------------------------------------------
    # 3. FATIAMENTO (CHUNKING) UNIFICADO
    # --------------------------------------------------------------------------
    print(f"\nFatiando um total de {len(documentos_totais)} documentos acumulados...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documentos_totais)
    print(f"Gerados {len(chunks)} chunks de contexto.")

    # --------------------------------------------------------------------------
    # 4. EMBEDDINGS E GRAVAÇÃO NO CHROMADB PERSISTENTE
    # --------------------------------------------------------------------------
    print(f"Gerando embeddings com '{MODEL_EMBEDDING}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_EMBEDDING,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print("Atualizando o banco vetorial local em './chroma_db'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    tempo_total = round(time.time() - tempo_inicio, 2)
    print(f"\nIngestão mista concluída com sucesso em {tempo_total}s!")

if __name__ == "__main__":
    executar_ingestao_mista()