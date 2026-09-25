# 1. Importa a função que sabe ler arquivos externos de configuração (.env)
from dotenv import load_dotenv
# 2. Importa a classe responsável por conectar o seu código ao serviço do OpenRouter
from langchain_openrouter import ChatOpenRouter

# 3. Procura o arquivo `.env` na raiz do projeto e carrega as variáveis dele (como a OPENROUTER_API_KEY)
#    Isso faz com que o Python "decore" a sua chave de acesso para usá-la secretamente depois
load_dotenv()

# 4. Cria e configura o seu "robô" pensante (objeto modelo)
#    Aqui você define que quer usar o OpenRouter e escolhe o catálogo de modelos gratuitos ("openrouter/free")
modelo = ChatOpenRouter(

   model="openrouter/free"

)

# 5. Envia a pergunta para a inteligência artificial através do comando `.invoke()`
#    O programa envia o texto para a internet, o OpenRouter processa e devolve um pac
resposta = modelo.invoke(
       "Explique o que é um modelo de linguagem em poucas palavras."

)

# 6. Exibe na tela apenas o texto puro gerado pela IA
#    O pacote 'resposta' vem com metadados (como tempo de resposta e tokens), mas o `.content` filtra e mostra só a mensagem final
print(resposta.content)