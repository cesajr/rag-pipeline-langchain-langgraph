# Pipeline RAG: da conexão básica a grafos, parâmetros e interface web

> **Aviso Importante:** Este repositório é mantido estritamente para **fins didáticos, de portfólio e evolução de estudos**. Ele serve como uma implementação de referência estruturada para o aprendizado de arquiteturas de *Retrieval-Augmented Generation* (RAG) e Engenharia de Agentes. **Contribuições externas e Pull Requests (PRs) não são aceitos.** Sinta-se à vontade para realizar um *fork* ou clonar para uso pessoal.

---

## Visão Geral do Projeto

Este repositório registra a jornada prática de desenvolvimento e maturação de um pipeline de RAG end-to-end utilizando ecossistemas modernos em **Python**. O projeto evolui progressivamente desde chamadas diretas a modelos de linguagem até grafos de estados complexos com memória persistente, filtragem avançada por limiares de similaridade, metadados, engenharia de prompts por personas e uma interface gráfica web reativa.

---

## Tech Stack e Arquitetura

* **Linguagem:** Python 3.14+
* **Orquestração & Agentes:** LangChain, LangGraph (`StateGraph`, `MemorySaver`, `add_messages`)
* **Banco Vetorial & Embeddings:** ChromaDB, HuggingFace Embeddings (`mixedbread-ai/mxbai-embed-large-v1`)
* **Data Loaders & Parsers:** `PyPDFLoader`, `BeautifulSoup` (`BSHTMLLoader`), `WebBaseLoader`, `WikipediaLoader`
* **Interface Gráfica Web:** Streamlit
* **Provedor de LLM:** OpenRouter API (`ChatOpenRouter` e `ChatOpenAI`)

---

## 📂 Estrutura do Repositório e Etapas Evolutivas

O código está organizado de forma modular e progressiva para demonstrar o domínio técnico em 8 marcos fundamentais:

```text
rag-pipeline-langchain-langgraph/
│
├── data/
│   ├── pagina.html
│   └── relatorio.pdf
│
├── src/
│   ├── example_research.py       # 1. Validação de conexão e chamada direta ao LLM
│   ├── mini_rag.py               # 2. Primeiro Mini-RAG funcional com Wikipedia & LangGraph
│   ├── mini_rag_wikipedia.py     # 3. Refatoração para integração nativa do OpenRouter
│   ├── mrag_wiki.py              # 4. RAG resiliente via Web Scraping com memória persistente
│   ├── test_loaders.py           # 5. Extração e padronização de multi-formatos (PDF e HTML)
│   ├── test_retrieval.py         # 6. Testes empíricos de busca e análise de ruído com top-k
│   ├── test_parameters.py        # 7. Ajuste fino de parâmetros (threshold, metadados e personas)
│   └── interface_rag.py          # 8. Aplicação web interativa de chatbot em Streamlit
│
├── .env.example                  # Modelo de variáveis de ambiente seguras
├── .gitignore                    # Regras de exclusão de arquivos sensíveis e caches
├── requirements.txt              # Gerenciador de dependências do projeto
└── README.md                     # Documentação oficial do portfólio
```

---

## Detalhamento Técnico dos Módulos

### 1. Conexão Inicial ao LLM (`src/example_research.py`)
* **Objetivo:** Validar a autenticação segura de credenciais via `python-dotenv` e testar a invocação direta do modelo gratuito na nuvem utilizando o wrapper `ChatOpenRouter`.

### 2. Primeiro Mini-RAG Funcional (`src/mini_rag.py`)
* **Objetivo:** Construir um fluxo fechado combinando ingestão de artigos da Wikipedia, divisão de texto em pedaços (*chunking* via `RecursiveCharacterTextSplitter`), vetorização local com HuggingFace, persistência no ChromaDB e orquestração sequencial de 2 nós com LangGraph.

### 3. Integração Nativa OpenRouter (`src/mini_rag_wikipedia.py`)
* **Objetivo:** Atualizar a arquitetura para utilizar o pacote dedicado `langchain-openrouter`, otimizando o roteamento dinâmico de modelos e o gerenciamento de headers HTTP.

### 4. RAG Resiliente com Memória de Conversação (`src/mrag_wiki.py`)
* **Objetivo:** Substituir loaders instáveis por raspagem web robusta (`WebBaseLoader`), introduzindo persistência de histórico por threads de conversação através do `MemorySaver` do LangGraph.

### 5. Ingestão e Padronização de Multi-Formatos (`src/test_loaders.py`)
* **Objetivo:** Demonstrar a capacidade de converter arquivos corporativos heterogêneos (relatórios PDF e páginas HTML brutas) no objeto universal e padronizado `Document` do LangChain, preservando metadados analíticos.

### 6. Validação de Recuperação & Comparação de Contexto (`src/test_retrieval.py`)
* **Objetivo:** Avaliar empiricamente a diferença qualitativa entre respostas puras de conhecimento geral do LLM vs. respostas ancoradas em contexto recuperado (*RAG*), analisando o impacto do parâmetro `top-k` na introdução de ruídos.

### 7. Ajuste Fino de Parâmetros e Prompts por Personas (`src/test_parameters.py`)
* **Objetivo:** Controlar a precisão da busca aplicando limiares estritos de similaridade matemática (`score_threshold`), filtros direcionados de metadados e testando engenharia de prompts estilísticos (ex: *Consultor Técnico* com baixa temperatura vs. *Mentor Criativo* com alta temperatura).

### 8. Interface Web Interativa (`src/interface_rag.py`)
* **Objetivo:** Empacotar toda a lógica em uma aplicação gráfica web reativa utilizando **Streamlit**, oferecendo entrada de texto em tempo real, estados visuais de carregamento (`st.spinner`) e tratamento de exceções amigável.

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
Obtenha sua chave de API gratuita no painel do [OpenRouter](https://openrouter.ai/).

Crie um arquivo chamado `.env` na raiz do projeto (utilizando o `.env.example` como base):
```env
OPENROUTER_API_KEY="sua-chave-aqui"
```

---

## Como Executar os Scripts e Aplicações

* **Para iniciar a interface web interativa (Streamlit):**
  ```bash
  streamlit run src/interface_rag.py
  ```
* **Para rodar os testes de parâmetros avançados e validações via terminal:**
  ```bash
  python src/test_parameters.py
  ```

---

## Segurança e Boas Práticas
* **Proteção de Segredos:** Credenciais e chaves de produção residem exclusivamente no arquivo local `.env`, que é bloqueado pelo `.gitignore` e nunca enviado ao repositório remoto.
* **Transparência de Modelos:** O arquivo `.env.example` fornece o gabarito das variáveis de ambiente necessárias para que outros desenvolvedores entendam a estrutura sem comprometer a segurança.
