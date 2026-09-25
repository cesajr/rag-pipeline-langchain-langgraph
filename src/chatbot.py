# 1. Módulo do Streamlit para criação da interface gráfica web
import streamlit as st

# 2. Módulos para manipulação do sistema e variáveis de ambiente
import os
from dotenv import load_dotenv

# 3. Módulo do LangChain para conexão com o OpenRouter
from langchain_openrouter import ChatOpenRouter

# Carrega a chave de API (OPENROUTER_API_KEY) declarada no arquivo .env
load_dotenv()

# Configuração da página na aba do navegador (Título da aba e Ícone)
st.set_page_config(page_title="RAG Chatbot", page_icon="💬")

# Título principal da interface
st.title("💬 RAG Chatbot")

# Subtítulo personalizado destacando as tecnologias utilizadas
st.subheader("Sistema de IA com LangChain + OpenRouter")

# Caixa de texto de múltiplas linhas para o usuário digitar a pergunta
pergunta = st.text_area(
    "Digite sua pergunta:", 
    placeholder="Ex: Explique em poucas palavras o que é LangChain e como ele ajuda no RAG."
)

# Botão de envio com rótulo personalizado
if st.button("Enviar"):
    # Valida se o usuário não enviou o campo em branco
    if not pergunta.strip():
        st.warning("Por favor, digite uma pergunta antes de clicar em enviar!")
    else:
        # Exibe o st.spinner enquanto o modelo processa a requisição
        with st.spinner("Carregando resposta..."):
            try:
                # Instancia o modelo de linguagem apontando para a API do OpenRouter
                modelo = ChatOpenRouter(
                    model="openrouter/free",
                    temperature=0.3
                )
                
                # Envia a pergunta do usuário para o LLM via LangChain
                resposta = modelo.invoke(pergunta)
                
                # Adiciona uma linha divisória estética na página
                st.divider()
                
                # Cabeçalho da área de exibição do resultado
                st.markdown("### 💡 Resposta:")
                
                # Renderiza o texto de resposta formatado na interface gráfica
                st.write(resposta.content)
                
            except Exception as e:
                # Trata eventuais falhas de conexão ou chave de API exibindo mensagem em vermelho
                st.error(f"Ocorreu um erro ao consultar a API: {e}")