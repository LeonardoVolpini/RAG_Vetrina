from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_pdfs(pdf_paths: list[str]) -> list[Document]:
    """
    Carica PDF con suddivisione ottimizzata per documenti tecnici di edilizia.
    """
    docs: list[Document] = []
    
    # Configurazione ottimizzata per documenti tecnici di edilizia
    # Chunks più grandi per catturare contesto completo di normative/specifiche
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,       # Chunks più grandi per mantenere il contesto delle normative
        chunk_overlap=350,     # Overlap maggiore per non perdere informazioni tra chunks
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Priorità di divisione su paragrafi
    )
    
    for path in pdf_paths:
        try:
            loader = PyPDFLoader(path)
            pages = loader.load()
            
            # Aggiungi metadati più dettagliati
            filename = os.path.basename(path)
            for i, page in enumerate(pages):
                page.metadata.update({
                    "source_pdf": filename,
                    "page": i + 1,
                    "total_pages": len(pages),
                    "file_path": path,
                })
            
            # Suddividi in chunks ottimizzati
            chunks = text_splitter.split_documents(pages)
            docs.extend(chunks)
        except Exception as e:
            print(f"Errore nel caricamento del PDF {path}: {str(e)}")
            
    return docs