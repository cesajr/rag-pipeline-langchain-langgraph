from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader

print("--- 1. CARREGANDO O ARQUIVO PDF ---")
try:
    loader_pdf = PyPDFLoader("relatorio.pdf")
    documentos_pdf = loader_pdf.load()
    print(f"Sucesso! Total de páginas no PDF: {len(documentos_pdf)}")
    
    if documentos_pdf:
        print("\n[Exemplo - PDF] Primeiros 300 caracteres:")
        print(documentos_pdf[0].page_content[:300])
        print("\n[Exemplo - PDF] Metadados:")
        print(documentos_pdf[0].metadata)
except Exception as e:
    print(f"Erro ao carregar o PDF: {e}")

print("\n" + "="*50 + "\n")

print("\n--- 2. CARREGANDO O ARQUIVO HTML ---")
try:
    # Lemos o arquivo HTML manualmente com utf-8 para evitar problemas de codificação
    with open("pagina.html", "r", encoding="utf-8", errors="ignore") as f:
        html_content = f.read()

    # Como o BSHTMLLoader lê caminhos de arquivo, podemos usar uma alternativa direta com BeautifulSoup
    from bs4 import BeautifulSoup
    from langchain_core.documents import Document

    soup = BeautifulSoup(html_content, "html.parser")
    texto_html = soup.get_text(separator="\n", strip=True)
    
    # Criamos o objeto Document manualmente no mesmo padrão do LangChain
    documentos_html = [
        Document(
            page_content=texto_html,
            metadata={"source": "pagina.html", "title": soup.title.string if soup.title else "Sem título"}
        )
    ]

    print(f"Sucesso! Total de documentos no HTML: {len(documentos_html)}")
    
    if documentos_html:
        print("\n[Exemplo - HTML] Primeiros 300 caracteres:")
        print(documentos_html[0].page_content[:300])
        print("\n[Exemplo - HTML] Metadados:")
        print(documentos_html[0].metadata)
except Exception as e:
    print(f"Erro ao carregar o HTML: {e}")