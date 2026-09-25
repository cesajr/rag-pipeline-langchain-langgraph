# 1. Módulos de ambiente e interface gráfica
import os
import streamlit as st
from dotenv import load_dotenv

# 2. Classes para estruturação das mensagens no LangChain
from langchain_core.messages import HumanMessage, AIMessage

# 3. Conexão oficial com o OpenRouter
from langchain_openrouter import ChatOpenRouter

# Carrega as variáveis declaradas no arquivo .env (como a OPENROUTER_API_KEY)
load_dotenv()

# Configuração da página na aba do navegador
st.set_page_config(page_title="Chatbot com Memória")

st.title("Chatbot com Memória de Conversação")
st.subheader("Mantendo o contexto histórico via LangChain + OpenRouter")

# ------------------------------------------------------------------
# ETAPA 1: Inicialização da Memória da Sessão no Streamlit
# ------------------------------------------------------------------
# O 'st.session_state' garante que a lista persista entre os recarregamentos da página
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ------------------------------------------------------------------
# Barra Lateral com Métricas e Botão para Limpar
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Opções do Chat")
    
    # Métrica: calcula quantos pares de Pergunta/Resposta existem na memória
    num_interacoes = len(st.session_state.chat_history) // 2
    st.metric(label="Interações Armazenadas", value=num_interacoes)
    
    st.divider()
    
    # Botão para redefinir o contexto e zerar o histórico
    if st.button("🗑️ Limpar Histórico de Conversa"):
        st.session_state.chat_history = []
        st.rerun()  # Atualiza a tela imediatamente


# ------------------------------------------------------------------
# ETAPA 2: Exibição do Histórico Anterior na Tela
# ------------------------------------------------------------------
# Garante que todas as mensagens anteriores fiquem visíveis na interface
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)


# ------------------------------------------------------------------
# ETAPA 3: Processamento da Nova Pergunta do Usuário
# ------------------------------------------------------------------
# Componente moderno de chat do Streamlit (campo de entrada fixo na parte inferior)
user_input = st.chat_input("Digite sua pergunta...")

if user_input:
    # 1. Adiciona a mensagem do usuário ao histórico da sessão
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    # Exibe a pergunta do usuário na tela imediatamente
    with st.chat_message("user"):
        st.write(user_input)

    # 2. Chama o LLM e envia o histórico acumulado de mensagens
    with st.chat_message("assistant"):
        with st.spinner("Consultando histórico e gerando resposta..."):
            try:
                # Instancia o modelo OpenRouter
                llm = ChatOpenRouter(
                    model="openrouter/free",
                    temperature=0.3
                )
                
                # Passa a lista completa 'st.session_state.chat_history' para o modelo
                response = llm.invoke(st.session_state.chat_history)
                
                # Renderiza a resposta gerada na interface
                st.write(response.content)
                
                # 3. Armazena a resposta da IA no histórico da sessão
                st.session_state.chat_history.append(AIMessage(content=response.content))
                
            except Exception as e:
                st.error(f"Erro ao processar a resposta: {e}")