# Pipeline RAG: Da Conexão Básica a Grafos, Busca Híbrida, Parâmetros e Interface Web

> **Aviso Importante:** Este repositório é mantido estritamente para **fins didáticos, de portfólio e evolução de estudos**. Ele serve como uma implementação de referência estruturada para o aprendizado de arquiteturas de *Retrieval-Augmented Generation* (RAG) e Engenharia de Agentes. **Contribuições externas e Pull Requests (PRs) não estão habilitados.** Sinta-se à vontade para realizar um *fork* ou clonar para uso pessoal.

---

## Visão Geral do Projeto

Este repositório registra a jornada prática de desenvolvimento e maturação de um pipeline de RAG *end-to-end* utilizando o ecossistema moderno de Python. O projeto evolui progressivamente desde chamadas diretas a modelos de linguagem até grafos de estados com memória persistente, vetorização em disco (ChromaDB), busca híbrida combinando acervo local (`data/`) e pesquisas ao vivo na internet (Wikipédia), filtragem dinâmica por limiares de similaridade (`score_threshold`), metadados e interfaces gráficas interativas em Streamlit com transparência e citação de fontes.

---

## Tech Stack e Arquitetura

* **Linguagem:** Python 3.14+
* **Orquestração e Agentes:** LangChain, LangGraph (`StateGraph`, `MemorySaver`, `add_messages`)
* **Banco Vetorial e Embeddings:** ChromaDB, HuggingFace Embeddings (`mixedbread-ai/mxbai-embed-large-v1`)
* **Data Loaders e Parsers:** `PyPDFLoader`, `BSHTMLLoader` (BeautifulSoup), `WebBaseLoader`, `WikipediaLoader`
* **Interface Gráfica Web:** Streamlit
* **Provedor de LLM:** OpenRouter API (`ChatOpenRouter` e `ChatOpenAI`) com suporte a modelos gratuitos (`openrouter/free`, `google/gemma-2-9b-it:free`, `meta-llama/llama-3.1-8b-instruct:free`)

---

## Estrutura do Repositório e Etapas Evolutivas

O código está organizado de forma modular e plana dentro do diretório `src/` para demonstrar a evolução cronológica do aprendizado:

```text
rag-pipeline-langchain-langgraph/
│
├── chroma_db/                    # Banco de dados vetorial local persistente (ignorado no Git)
│
├── data/                         # Diretório de documentos locais de entrada (PDFs e HTMLs)
│   ├── pagina.html
│   └── relatorio.pdf
│
├── src/                          # Módulos e aplicações em ordem evolutiva
│   ├── example_research.py       # 1. Validação de conexão e chamada direta ao LLM
│   ├── mini_rag.py               # 2. Primeiro Mini-RAG funcional com Wikipedia e LangGraph
│   ├── mini_rag_wikipedia.py     # 3. Refatoração para integração nativa do OpenRouter
│   ├── mrag_wiki.py              # 4. RAG resiliente via Web Scraping com memória persistente
│   ├── test_loaders.py           # 5. Extração e padronização de multi-formatos (PDF e HTML)
│   ├── test_retrieval.py         # 6. Testes empíricos de busca e análise de ruído com top-k
│   ├── test_parameters.py        # 7. Ajuste fino de parâmetros (threshold, metadados e personas)
│   ├── chatbot.py                # 8. Interface web de Chatbot com Memória de Conversação
│   ├── chatbot_rag.py            # 9. Interface web reativa básica de RAG
│   ├── ingest.py                 # 10. Processamento de PDFs/URLs e criação do ChromaDB persistente
│   └── app.py                    # 11. Aplicação RAG Híbrida final (Local + Web ao Vivo + Citações)
│
├── .env                          # Variáveis de ambiente locais (não versionado)
├── .env.example                  # Modelo de variáveis de ambiente seguras
├── .gitignore                    # Regras de exclusão de arquivos sensíveis, caches e bancos
├── requirements.txt              # Dependências do projeto
└── README.md                     # Documentação oficial do portfólio
```

---

## Detalhamento Técnico dos Módulos

### 1. Conexão Inicial ao LLM (`src/example_research.py`)
* **Objetivo:** Validar a autenticação segura de credenciais via `python-dotenv` e testar a invocação direta do modelo na nuvem utilizando o wrapper `ChatOpenRouter`.

### 2. Primeiro Mini-RAG Funcional (`src/mini_rag.py`)
* **Objetivo:** Construir um fluxo fechado combinando ingestão de artigos da Wikipedia, divisão de texto em trechos (`RecursiveCharacterTextSplitter`), vetorização local com HuggingFace, persistência no ChromaDB e orquestração sequencial de 2 nós com LangGraph.

### 3. Integração Nativa OpenRouter (`src/mini_rag_wikipedia.py`)
* **Objetivo:** Atualizar a arquitetura para utilizar o pacote dedicado `langchain-openrouter`, otimizando o roteamento dinâmico de modelos e o gerenciamento de cabeçalhos HTTP.

### 4. RAG Resiliente com Memória (`src/mrag_wiki.py`)
* **Objetivo:** Substituir loaders instáveis por raspagem web robusta (`WebBaseLoader`), introduzindo persistência de histórico por *threads* de conversação através do `MemorySaver` do LangGraph.

### 5. Ingestão e Padronização Multi-Formato (`src/test_loaders.py`)
* **Objetivo:** Demonstrar a capacidade de converter arquivos corporativos heterogêneos (relatórios PDF e páginas HTML brutas) no objeto universal e padronizado `Document` do LangChain, preservando metadados analíticos.

### 6. Validação de Recuperação e Comparação de Contexto (`src/test_retrieval.py`)
* **Objetivo:** Avaliar empiricamente a diferença qualitativa entre respostas puras de conhecimento geral do LLM vs. respostas ancoradas em contexto recuperado (RAG), analisando o impacto do parâmetro `top-k` na introdução de ruídos.

### 7. Ajuste Fino de Parâmetros e Prompts por Personas (`src/test_parameters.py`)
* **Objetivo:** Controlar a precisão da busca aplicando limiares estritos de similaridade matemática (`score_threshold`), filtros direcionados de metadados e testando engenharia de *prompts* estilísticos (ex.: Consultor Técnico vs. Mentor Criativo).

### 8. Chatbot com Memória de Conversação (`src/chatbot.py`)
* **Objetivo:** Implementar uma interface interativa em Streamlit que mantém e reutiliza o histórico do diálogo (`HumanMessage` e `AIMessage`) via `st.session_state`, permitindo acompanhar o contexto e redefinir a memória.

### 9. Aplicação RAG Web Básica (`src/chatbot_rag.py`)
* **Objetivo:** Unificar o pipeline de recuperação semântica no banco vetorial com uma interface gráfica reativa em Streamlit, oferecendo busca de documentos, feedback de carregamento em tempo real (`st.spinner`) e geração de respostas fundamentadas.

### 10. Ingestão Persistente de Dados (`src/ingest.py`)
* **Objetivo:** Criar um pipeline de ingestão desacoplado que lê documentos locais (PDFs em `./data`) e URLs web, gera *embeddings* vetorizados usando `mixedbread-ai/mxbai-embed-large-v1` e armazena os vetores em disco no diretório `./chroma_db`.

### 11. Aplicação RAG Híbrida Completa (`src/app.py`)
* **Objetivo:** Construir a interface web de produção contendo:
  * **Busca Híbrida Simultânea:** Consulta o ChromaDB local e pesquisa na internet em tempo real via `WikipediaQueryRun`.
  * **Otimização de Performance:** Uso de `@st.cache_resource` para manter *embeddings* e conexão com banco vetorial em memória RAM.
  * **Transparência de Fontes:** Painel expansível (`st.expander`) detalhando os trechos recuperados, arquivos de origem e números de páginas.
  * **Guardrails Anti-Alucinação:** *Prompts* de sistema instruindo o modelo a admitir explicitamente a ausência de dados quando o contexto não contiver a resposta.

---

## Guia de Instalação e Execução

### 1. Clonar o Repositório

```bash
git clone https://github.com/cesajr/rag-pipeline-langchain-langgraph.git
cd rag-pipeline-langchain-langgraph
```

### 2. Configurar o Ambiente Virtual

```bash
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar as Chaves de Acesso

1. Obtenha sua chave de API gratuita no painel do OpenRouter.
2. Crie um arquivo chamado `.env` na raiz do projeto (utilizando o arquivo `.env.example` como referência):

```env
OPENROUTER_API_KEY="sua-chave-aqui"
```

### Como Executar as Aplicações

**Ingestão Inicial dos Documentos Locais (PDFs e Web):**

```bash
python src/ingest.py
```
*Este comando criará o banco vetorial na pasta `./chroma_db`.*

**Iniciar a Aplicação RAG Híbrida Principal (Local + Web):**

```bash
streamlit run src/app.py
```

**Iniciar o Chatbot com Memória de Conversa:**

```bash
streamlit run src/chatbot.py
```

**Rodar os Testes de Parâmetros Avançados no Terminal:**

```bash
python src/test_parameters.py
```

---

## Segurança e Boas Práticas

* **Proteção de Segredos:** Credenciais e chaves de produção residem exclusivamente no arquivo local `.env`, que é bloqueado pelo `.gitignore` e nunca enviado ao repositório remoto.
* **Isolamento de Dados Locais:** A pasta `./chroma_db` contendo os índices vetoriais é mantida fora do versionamento para garantir a leveza e segurança do repositório.
* **Transparência de Configuração:** O arquivo `.env.example` fornece o gabarito das variáveis de ambiente necessárias para a execução imediata por outros desenvolvedores.