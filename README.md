# Jornada Prática de RAG: De Chamadas Básicas a Grafos com Memória e Loaders

> **Aviso Importante:**  
> Este repositório foi construído para fins estritamente **didáticos e educacionais**, funcionando como um guia de estudo e referência de código passo a passo. **Este projeto é mantido como um arquivo de estudo individual e NÃO aceita contribuições externas, alterações ou Pull Requests (PRs).** Sinta-se à vontade para realizar o *fork* ou clonar o projeto para uso pessoal e de estudo.

---

Este repositório registra a evolução prática da construção de pipelines de **Retrieval-Augmented Generation (RAG)** e Engenharia de Agentes com **Python**, **LangChain**, **LangGraph**, **ChromaDB** e **OpenRouter**.

O objetivo deste portfólio é demonstrar, de forma incremental e didática, a maturidade no manuseio de dados não estruturados, bancos vetoriais, controle de estado de conversação e boas práticas de segurança.

---

## Stacks & Libs
* **Linguagem:** Python 3.10+
* **Orquestração & Agentes:** LangChain, LangGraph (`StateGraph`, `MemorySaver`)
* **Vector Database & Embeddings:** ChromaDB, HuggingFace (`mixedbread-ai/mxbai-embed-large-v1`)
* **Data Loaders:** `PyPDFLoader`, `BSHTMLLoader` / `BeautifulSoup`, `WebBaseLoader`, `WikipediaLoader`
* **LLM Provider:** OpenRouter API (`ChatOpenRouter` e `ChatOpenAI`)

---

## Arquitetura Evolutiva do Projeto

Abaixo está o detalhamento de cada script criado ao longo da curva de aprendizado:

### 1. Chamada Inicial ao LLM (`src/example_research.py`)
> **Objetivo:** Validar a conexão com a API do OpenRouter e a autenticação via variáveis de ambiente.

* **O que faz:** Lê a chave de API de forma segura usando `python-dotenv` e faz uma requisição direta ao modelo gratuito `openrouter/free` via `ChatOpenRouter`.
* **Conceito-chave:** Abstração de chamadas a modelos de linguagem e filtragem do conteúdo bruto da resposta via `resposta.content`.

---

### 2. Primeiro Mini-RAG Funcional com Wikipedia (`src/mini_rag.py`)
> **Objetivo:** Construir um pipeline RAG fim a fim alimentado por busca semântica em tempo real na Wikipedia.

* **O que faz:**
  1. **Coleta:** Baixa artigos sobre "Inteligência Artificial" e "História da Internet" via `WikipediaLoader`.
  2. **Chunking:** Fatia os textos usando `RecursiveCharacterTextSplitter` (1000 caracteres, overlap de 200).
  3. **Embedding & Vetores:** Gera vetores numéricos locais (`mxbai-embed-large-v1`) e salva no `ChromaDB`.
  4. **Orquestração LangGraph:** Define um `StateGraph` de dois nós (`buscar_contexto` ➔ `gerar_resposta`).
  5. **Interface:** Executa um loop interativo via terminal para o usuário fazer perguntas livres.

---

### 3. Refatoração para Integração Oficial OpenRouter (`src/mini_rag_wikipedia.py`)
> **Objetivo:** Refatorar a conexão do LLM para o pacote dedicado `langchain-openrouter`.

* **O que faz:** Substitui a chamada genérica `ChatOpenAI(base_url=...)` pela classe nativa `ChatOpenRouter(model="openrouter/free")`.
* **Conceito-chave:** Redução de código boilerplate, maior resiliência no roteamento automático de modelos gratuitos e gerenciamento nativo de headers de API.

---

### 4. RAG Resiliente com Memória de Conversação (`src/mrag_wiki.py`)
> **Objetivo:** Resolver problemas de instabilidade em APIs de terceiros e adicionar retenção de contexto entre perguntas.

* **O que faz:**
  * **Ingestão Robusta via Web:** Substitui o `WikipediaLoader` frágil por `WebBaseLoader`, raspando diretamente as URLs oficiais e adicionando estratégias de *fallback* (contingência).
  * **Memória Persistente (`MemorySaver`):** Atualiza o `AgentState` com `Annotated[list, add_messages]`.
  * **Threads de Conversa:** Usa `thread_id` para permitir perguntas implícitas (ex: *"Quais foram os principais nomes desse marco?"* sem precisar repetir o assunto).

---

### 5. Ingestão e Padronização de Multi-Formatos (`src/test_loaders.py`)
> **Objetivo:** Demonstrar a capacidade de extração e padronização de documentos não estruturados corporativos (PDFs e HTMLs).

* **O que faz:**
  * **Leitura de PDFs:** Usa `PyPDFLoader` (`pypdf`) para fatiar um relatório corporativo página a página, capturando metadados avançados (`total_pages`, `producer`, `creationdate`).
  * **Parsing de HTML Web:** Processa páginas HTML usando `BeautifulSoup`, tratando problemas de codificação (`utf-8`) e extraindo o título original da aba (`title`) e origem (`source`).
* **Conceito-chave:** Prova prática de que qualquer arquivo bruto é convertido no padrão universal `Document` do LangChain.

---

## ⚙️ Como Configurar e Executar

### 1. Clonar o Repositório
```bash
git clone https://github.com/cesajr/rag-pipeline-langchain-langgraph.git
cd rag-pipeline-langchain-langgraph
```

### 2. Criar e Ativar o Ambiente Virtual
```bash
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 4. Configurar Variáveis de Ambiente
Crie a sua chave de API gratuitamente no painel do [OpenRouter](https://openrouter.ai/). 

Em seguida, crie um arquivo `.env` na raiz do projeto (baseado no `.env.example`):
```env
OPENROUTER_API_KEY="sk-or-v1-sua-chave-aqui"
```

### 5. Executar os Scripts
* **Para testar o RAG com Memória e Web Scraping:**
  ```bash
  python src/mrag_wiki.py
  ```
* **Para testar os Loaders de PDF e HTML:**
  ```bash
  python src/test_loaders.py
  ```

---

## Segurança e Boas Práticas
* O arquivo `.env` contendo as credenciais de produção está explicitamente listado no `.gitignore` e **nunca** é enviado ao controle de versão.
* Inclusão do `.env.example` para documentar os pré-requisitos de execução do projeto de forma transparente sem expor chaves sensíveis.
