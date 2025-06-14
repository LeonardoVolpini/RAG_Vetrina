from typing import List, Dict, Any
import json
import os
from .config import settings

class FewShotExampleManager:
    """Gestisce gli esempi few-shot per il sistema RAG con supporto per reasoning step-by-step"""
    
    def __init__(self, examples_base_dir: str = None):
        self.examples_base_dir = examples_base_dir or os.path.join(os.path.dirname(settings.VECTOR_STORE_PATH), "few_shot_examples")
        self._ensure_examples_directory()
    
    def _ensure_examples_directory(self):
        """Crea la directory degli esempi se non esiste"""
        os.makedirs(self.examples_base_dir, exist_ok=True)
    
    def _get_examples_file_path(self, name: bool, description: bool) -> str:
        """Determina il percorso del file JSON basato sui parametri name e description"""
        if name is None and description is None:
            filename = "single_product.json"
        elif name and description:
            filename = "name_description_image.json"
        elif not name and description:
            filename = "description_image.json"
        elif name and not description:
            filename = "name_image.json"
        else:
            filename = "image.json"
            
        print(f"Utilizzo file esempi: {filename})")
        
        return os.path.join(self.examples_base_dir, filename)
    
    def _load_examples(self, name: bool, description: bool) -> List[Dict[str, str]]:
        """Carica gli esempi dal file JSON appropriato"""
        examples_file = self._get_examples_file_path(name, description)
        
        try:
            if os.path.exists(examples_file):
                with open(examples_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Restituisce lista vuota se il file non esiste
                return []
        except Exception as e:
            print(f"Errore nel caricamento degli esempi da {examples_file}: {str(e)}")
            return []
    
    def _save_examples(self, examples: List[Dict[str, str]], name: bool, description: bool):
        """Salva gli esempi nel file JSON appropriato"""
        examples_file = self._get_examples_file_path(name, description)
        
        try:
            with open(examples_file, 'w', encoding='utf-8') as f:
                json.dump(examples, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Errore nel salvare gli esempi in {examples_file}: {str(e)}")
    
    def add_example(self, question: str, answer: str, context_snapshot: str, reasoning: str, name: bool, description: bool):
        """Aggiunge un nuovo esempio nel file appropriato"""
        
        # Carica gli esempi esistenti
        existing_examples = self._load_examples(name, description)
        
        new_example = {
            "context_snapshot": context_snapshot.strip(),
            "question": question.strip(),
            "answer": answer.strip(),
            "reasoning": reasoning.strip()
        }
        
        existing_examples.append(new_example)
        self._save_examples(existing_examples, name, description)

        filename = self._get_examples_file_path(name, description).split('/')[-1]
        print(f"Aggiunto nuovo esempio in {filename}. Totale esempi: {len(existing_examples)}")
    
    def remove_example(self, index: int, name: bool, description: bool):
        """Rimuove un esempio per indice dal file appropriato"""
        examples = self._load_examples(name, description)
        
        if 0 <= index < len(examples):
            removed = examples.pop(index)
            self._save_examples(examples, name, description)
            filename = self._get_examples_file_path(name, description).split('/')[-1]
            print(f"Rimosso esempio da {filename}: {removed['question']}")
            return True
        return False
    
    def get_examples(self, name: bool, description: bool, max_examples: int = None) -> List[Dict[str, str]]:
        """Restituisce gli esempi dal file appropriato (limitati se specificato)"""
        examples = self._load_examples(name, description)
        
        if max_examples:
            return examples[:max_examples]
        return examples
    
    def _format_context_for_prompt(self, context_snapshot: str) -> str:
        """
        Formatta il context_snapshot per il template del prompt.
        Gestisce sia stringhe JSON che stringhe normali e escapa le parentesi graffe.
        """
        if not context_snapshot or context_snapshot.strip() == "":
            return ""
            
        try:
            # Prova a parsare come JSON
            parsed_context = json.loads(context_snapshot)
            # Se è un oggetto o array, formattalo in modo leggibile
            if isinstance(parsed_context, (dict, list)):
                formatted_json = json.dumps(parsed_context, indent=2, ensure_ascii=False)
                # Escapa le parentesi graffe per evitare conflitti con il template
                return formatted_json.replace("{", "{{").replace("}", "}}")
            else:
                # Se è un valore semplice, restituiscilo come stringa
                result = str(parsed_context)
                return result.replace("{", "{{").replace("}", "}}")
        except (json.JSONDecodeError, TypeError):
            # Se non è JSON valido, restituisci la stringa originale
            result = context_snapshot.strip()
            return result.replace("{", "{{").replace("}", "}}")
    
    def format_examples_for_prompt(self, name: bool, description: bool, max_examples: int = 3) -> str:
        """
        Formatta gli esempi per il prompt dal file appropriato
        
        Args:
            name: Se includere esempi per richieste di name
            description: Se includere esempi per richieste di description  
            max_examples: Numero massimo di esempi da includere
        """
        examples = self.get_examples(name, description, max_examples)
        if not examples:
            return ""
            
        formatted_examples = []
        
        for i, example in enumerate(examples, 1):
            # Salta esempi vuoti
            if not example.get('question') or not example.get('answer'):
                continue
                
            # Formatta il contesto in modo sicuro
            context_for_prompt = self._format_context_for_prompt(example.get('context_snapshot', ''))
            
            formatted_example = f"""Esempio {i}:
Contesto: {context_for_prompt}
Domanda: {example['question']}
Processo di ragionamento: {example.get('reasoning', '')}
Risposta: {example['answer']}"""
            
            formatted_examples.append(formatted_example)
        
        return "\n\n".join(formatted_examples)
    
    def get_relevant_examples(self, query: str, name: bool, description: bool, max_examples: int = 3) -> str:
        """
        Recupera gli esempi più rilevanti per una query specifica usando similarity search
        
        Args:
            query: Query dell'utente
            name: Se la query richiede informazioni sui nomi
            description: Se la query richiede informazioni sulle descrizioni
            store: Vector store per il similarity search
            max_examples: Numero massimo di esempi da restituire
        """
        examples = self._load_examples(name, description)
        
        if not examples:
            return ""
        
        # Filtra esempi vuoti
        valid_examples = [ex for ex in examples if ex.get('question') and ex.get('answer')]
        if not valid_examples:
            return ""
        
        try:
            # Crea un mini-vector store con le query degli esempi
            from langchain.docstore.document import Document
            from .embeddings import get_embeddings
            from langchain_community.vectorstores import FAISS
            
            # Crea documenti dalle query degli esempi
            example_docs = []
            for idx, example in enumerate(valid_examples):
                doc = Document(
                    page_content=example['question'],
                    metadata={'example_index': idx}
                )
                example_docs.append(doc)
            
            if not example_docs:
                return ""
            
            # Crea un vector store temporaneo per gli esempi
            embeddings = get_embeddings()   # Usa lo stesso provider degli embeddings principali
            examples_store = FAISS.from_documents(example_docs, embeddings)
            
            # Trova gli esempi più simili alla query
            similar_examples = examples_store.max_marginal_relevance_search(
                query, 
                k=min(max_examples, len(example_docs)), 
                fetch_k=len(example_docs)
            )
            
            formatted_examples = []
            for i, sim_doc in enumerate(similar_examples, 1):
                example_idx = sim_doc.metadata['example_index']
                example = valid_examples[example_idx]
                
                # Formatta il contesto in modo sicuro
                context_for_prompt = self._format_context_for_prompt(example.get('context_snapshot', ''))
                
                formatted_example = f"""Esempio {i}:
Contesto: {context_for_prompt}
Domanda: {example['question']}
Processo di ragionamento: {example.get('reasoning', '')}
Risposta: {example['answer']}"""
                
                formatted_examples.append(formatted_example)
            
            return "\n\n".join(formatted_examples)
            
        except Exception as e:
            print(f"Errore nel recupero degli esempi rilevanti: {str(e)}")
            # Fallback agli esempi standard
            return self.format_examples_for_prompt(name, description, max_examples)
    
    def list_all_example_files(self) -> Dict[str, int]:
        """Restituisce un dizionario con tutti i file di esempi e il numero di esempi contenuti"""
        file_counts = {}
        
        combinations = [
            (True, True, "name_description_image.json"),
            (False, True, "description_image.json"), 
            (True, False, "name_image.json"),
            (False, False, "image.json"),
            (None, None, "single_product.json")
        ]
        
        for name, desc, filename in combinations:
            examples = self._load_examples(name, desc)
            file_counts[filename] = len(examples)
        
        return file_counts