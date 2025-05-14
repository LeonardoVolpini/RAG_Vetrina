from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
import os

def load_pdfs(pdf_paths: list[str]) -> list[Document]:
    """
    Carica PDF e suddivide in pagine come Document LangChain.
    """
    docs: list[Document] = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages = loader.load_and_split()
        for page in pages:
            page.metadata["source_pdf"] = os.path.basename(path)
            docs.append(page)
    return docs
