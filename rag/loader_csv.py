from langchain.docstore.document import Document
import pandas as pd
import os
from typing import List, Dict, Any, Optional

def load_csv(file_path: str, header_row: int = 0, include_columns: Optional[List[str]] = None) -> List[Document]:
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
    
    try:
        # Carica il file con pandas
        if extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, header=header_row)
        else:  # Assume CSV per default o altro formato tabellare
            df = pd.read_csv(file_path, header=header_row)
        
        # Filtra colonne se specificato
        if include_columns:
            df = df[include_columns]
        
        docs = []
        
        # Approccio 1: Un documento per riga con metadati
        for index, row in df.iterrows():
            # Converti la riga in testo
            row_text = " ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            
            metadata = {
                "source_file": filename,
                "file_path": file_path,
                "row_index": index,
                "content_type": "tabular_data"
            }
            
            # Aggiungi valori di colonne chiave nei metadati per migliorare la ricerca
            for col, val in row.items():
                if pd.notna(val):
                    safe_col_name = str(col).replace('.', '_').replace(' ', '_').lower()
                    metadata[f"col_{safe_col_name}"] = str(val)
            
            docs.append(Document(page_content=row_text, metadata=metadata))
        
        # Approccio 2: Un documento con panoramica statistica
        if len(df) > 0:
            # Crea una panoramica statistica per l'intero dataset
            summary_parts = []
            summary_parts.append(f"Sommario del file {filename}")
            summary_parts.append(f"Numero totale di righe: {len(df)}")
            summary_parts.append(f"Colonne: {', '.join(df.columns.tolist())}")
            
            # Aggiungi informazioni statistiche per colonne numeriche
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                summary_parts.append("\nStatistiche per colonne numeriche:")
                for col in numeric_cols:
                    if df[col].notna().any():
                        summary_parts.append(f"  - {col}: min={df[col].min()}, max={df[col].max()}, media={df[col].mean():.2f}, mediana={df[col].median()}")
            
            # Crea un documento con il sommario
            summary_text = "\n".join(summary_parts)
            docs.append(Document(
                page_content=summary_text,
                metadata={
                    "source_file": filename,
                    "file_path": file_path,
                    "content_type": "tabular_summary",
                    "total_rows": len(df),
                    "total_columns": len(df.columns)
                }
            ))
        
        return docs
        
    except Exception as e:
        print(f"Errore nel caricamento del file {file_path}: {str(e)}")
        return []