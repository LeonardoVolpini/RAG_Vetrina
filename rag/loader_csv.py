import os
from langchain.docstore.document import Document
import pandas as pd
from typing import List, Optional

def extract_brand_from_filename(filename: str) -> str:
    """
    Estrae la marca dal nome del file (prima parte prima di '_')
    Es: 'yamato_products.csv' -> 'products'
    """
    basename = os.path.splitext(filename)[0]  # Rimuove estensione
    brand = basename.split('_')[0].lower().strip()
    return brand

def load_csv(file_path: str, header_row: int = 0, 
                                include_columns: Optional[List[str]] = None) -> List[Document]:
    """
    Carica un file CSV/Excel e lo converte in documenti utilizzabili per RAG.
    
    Args:
        file_path: Percorso del file CSV/Excel
        header_row: Riga dell'intestazione (default: 0)
        include_columns: Lista di colonne da includere (se None, include tutte)
    
    Returns:
        Lista di Document pronti per l'embedding
    """
    filename = os.path.basename(file_path)
    extension = os.path.splitext(filename)[1].lower()
    
    # Estrai la marca dal nome del file
    brand = extract_brand_from_filename(filename)
    
    try:
        # Carica il file
        if extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, header=header_row)
        else:
            df = pd.read_csv(file_path, header=header_row)
        
        if include_columns:
            df = df[include_columns]
        
        docs = []
        
        # Crea un documento per ogni riga con metadati di marca
        for index, row in df.iterrows():
            # Testo della riga
            row_text = " ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            
            metadata = {
                "source_file": filename,
                "file_path": file_path,
                "row_index": index,
                "content_type": "tabular_data",
                "brand": brand,  # Metadato chiave per il filtering
                "brand_normalized": brand.lower().replace(' ', '_')
            }
            
            # Aggiungi valori delle colonne nei metadati
            for col, val in row.items():
                if pd.notna(val):
                    safe_col_name = str(col).replace('.', '_').replace(' ', '_').lower()
                    metadata[f"col_{safe_col_name}"] = str(val)
            
            docs.append(Document(page_content=row_text, metadata=metadata))
        
        return docs
        
    except Exception as e:
        print(f"Errore nel caricamento del file {file_path}: {str(e)}")
        return []