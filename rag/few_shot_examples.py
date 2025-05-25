from typing import List, Dict, Any
import json
import os
from .config import settings

class FewShotExampleManager:
    """Gestisce gli esempi few-shot per il sistema RAG"""
    
    def __init__(self, examples_file: str = None):
        self.examples_file = examples_file or os.path.join(os.path.dirname(settings.VECTOR_STORE_PATH), "few_shot_examples.json")
        self.examples = self._load_examples()
    
    def _load_examples(self) -> List[Dict[str, str]]:
        """Carica gli esempi da file JSON"""
        try:
            if os.path.exists(self.examples_file):
                with open(self.examples_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Crea esempi di default
                default_examples = self._get_default_examples()
                self._save_examples(default_examples)
                return default_examples
        except Exception as e:
            print(f"Errore nel caricamento degli esempi: {str(e)}")
            return self._get_default_examples()
    
    def _save_examples(self, examples: List[Dict[str, str]]):
        """Salva gli esempi su file JSON"""
        try:
            os.makedirs(os.path.dirname(self.examples_file), exist_ok=True)
            with open(self.examples_file, 'w', encoding='utf-8') as f:
                json.dump(examples, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Errore nel salvare gli esempi: {str(e)}")
    
    def _get_default_examples(self) -> List[Dict[str, str]]:
        """Esempi di default per il dominio edilizia (solo query-answer)"""
        return [
            {
                "question": "Descrivimi il cemento Portland CEM II/A-LL 42.5 R",
                "answer": "Il cemento Portland CEM II/A-LL 42.5 R è un cemento con aggiunta di calcari che raggiunge una resistenza caratteristica di 42.5 MPa a 28 giorni. La sigla 'R' indica un rapido indurimento, rendendolo particolarmente adatto per applicazioni che richiedono sviluppo precoce della resistenza. È conforme alla norma EN 197-1 e viene utilizzato principalmente per calcestruzzo strutturale e nella produzione di prefabbricati."
            },
            {
                "question": "Che caratteristiche ha il trapano GSB 13 RE?",
                "answer": "Il GSB 13 RE è un trapano a percussione con potenza di 600W e mandrino da 13mm. Ha velocità variabile fino a 2800 giri/min e coppia massima di 34 Nm. Il peso contenuto di 1.8 kg lo rende maneggevole per lavori prolungati. È adatto per foratura in muratura, legno e metallo grazie alla funzione percussione."
            },
            {
                "question": "Descrivimi l'isolante termico disponibile",
                "answer": "Esistono più possibili corrispondenze per isolante termico."
            },
            {
                "question": "Che tipo di intonaco è disponibile?",
                "answer": "Esistono più possibili corrispondenze per intonaco."
            },
            {
                "question": "Dimmi le specifiche del prodotto XYZ123 che non esiste",
                "answer": "Non lo so."
            }
        ]
    
    def add_example(self, question: str, answer: str):
        """Aggiunge un nuovo esempio (solo question e answer)"""
        new_example = {
            "question": question.strip(),
            "answer": answer.strip()
        }
        self.examples.append(new_example)
        self._save_examples(self.examples)
        print(f"Aggiunto nuovo esempio. Totale esempi: {len(self.examples)}")
    
    def remove_example(self, index: int):
        """Rimuove un esempio per indice"""
        if 0 <= index < len(self.examples):
            removed = self.examples.pop(index)
            self._save_examples(self.examples)
            print(f"Rimosso esempio: {removed['question']}")
            return True
        return False
    
    def get_examples(self, max_examples: int = None) -> List[Dict[str, str]]:
        """Restituisce gli esempi (limitati se specificato)"""
        if max_examples:
            return self.examples[:max_examples]
        return self.examples
    
    def format_examples_for_prompt(self, max_examples: int = 3, store=None) -> str:
        """
        Formatta gli esempi per il prompt, opzionalmente recuperando il contesto dal vector store
        
        Args:
            max_examples: Numero massimo di esempi da includere
            store: Vector store FAISS per recuperare il contesto (opzionale)
        """
        examples = self.get_examples(max_examples)
        formatted_examples = []
        
        for i, example in enumerate(examples, 1):
            if store:
                # Recupera il contesto dal vector store per questa query
                try:
                    # Cerca documenti rilevanti per la query dell'esempio
                    relevant_docs = store.similarity_search(example['question'], k=2)
                    context = "\n".join([doc.page_content for doc in relevant_docs])
                    
                    formatted_example = f"""
                        Esempio {i}:
                        Contesto recuperato: {context[:500]}{'...' if len(context) > 500 else ''}
                        Domanda: {example['question']}
                        Risposta: {example['answer']}
                        """
                except Exception as e:
                    print(f"Errore nel recupero del contesto per l'esempio {i}: {str(e)}")
                    # Fallback senza contesto
                    formatted_example = f"""
                        Esempio {i}:
                        Domanda: {example['question']}
                        Risposta: {example['answer']}
                        """
            else:
                # Formato senza contesto
                formatted_example = f"""
                    Esempio {i}:
                    Domanda: {example['question']}
                    Risposta: {example['answer']}
                    """
            
            formatted_examples.append(formatted_example)
        
        return "\n".join(formatted_examples)
    
    def get_relevant_examples(self, query: str, store, max_examples: int = 3) -> str:
        """
        Recupera gli esempi più rilevanti per una query specifica usando similarity search
        
        Args:
            query: Query dell'utente
            store: Vector store per il similarity search
            max_examples: Numero massimo di esempi da restituire
        """
        if not self.examples:
            return ""
        
        try:
            # Crea un mini-vector store con le query degli esempi
            from langchain.docstore.document import Document
            from .embeddings import get_embeddings
            from langchain_community.vectorstores import FAISS
            
            # Crea documenti dalle query degli esempi
            example_docs = []
            for idx, example in enumerate(self.examples):
                doc = Document(
                    page_content=example['question'],
                    metadata={'example_index': idx}
                )
                example_docs.append(doc)
            
            if not example_docs:
                return ""
            
            # Crea un vector store temporaneo per gli esempi
            embeddings = get_embeddings()  # Usa lo stesso provider degli embeddings principali
            examples_store = FAISS.from_documents(example_docs, embeddings)
            
            # Trova gli esempi più simili alla query
            similar_examples = examples_store.similarity_search(query, k=max_examples)
            
            formatted_examples = []
            for i, sim_doc in enumerate(similar_examples, 1):
                example_idx = sim_doc.metadata['example_index']
                example = self.examples[example_idx]
                
                # Recupera il contesto dal vector store principale per questo esempio
                try:
                    relevant_docs = store.similarity_search(example['question'], k=2)
                    context = "\n".join([doc.page_content for doc in relevant_docs])
                    
                    formatted_example = f"""
                        Esempio {i}:
                        Contesto: {context[:400]}{'...' if len(context) > 400 else ''}
                        Domanda: {example['question']}
                        Risposta: {example['answer']}
                        """
                except Exception as e:
                    print(f"Errore nel recupero del contesto per l'esempio rilevante {i}: {str(e)}")
                    formatted_example = f"""
                        Esempio {i}:
                        Domanda: {example['question']}
                        Risposta: {example['answer']}
                        """
                
                formatted_examples.append(formatted_example)
            
            return "\n".join(formatted_examples)
            
        except Exception as e:
            print(f"Errore nel recupero degli esempi rilevanti: {str(e)}")
            # Fallback agli esempi standard
            return self.format_examples_for_prompt(max_examples, store)