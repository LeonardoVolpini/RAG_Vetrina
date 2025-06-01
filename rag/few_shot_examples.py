from typing import List, Dict, Any
import json
import os
from .config import settings

class FewShotExampleManager:
    """Gestisce gli esempi few-shot per il sistema RAG con supporto per reasoning step-by-step"""
    
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
        """Esempi di default con reasoning step-by-step"""
        return [
            {
                "context_snapshot": """[ 
                                        { "name": "CARTA ABRASIVA PER MOLLA - FDS 140. Abrasive Paper- Clamp Type for FDS 140", "brand": "AEG", "description": "Carta abrasiva preforata per legno, vernice, smalto e spatolato; Tipo di grana: ossido di alluminio legato con resina artificiale su carta speciale antistrappo; Ideale per legno e lavori di carrozzeria", "image_url": "/images/aeg_products/CARTA_ABRASIVA_PER_MOLLA_-_FDS_140_Abrasive_Paper-_Clamp_Type_for_FDS_140.webp"},
                                        { "name": "Martello demolitore 7 kg SDS-MAX MH 7E", "brand": "AEG", "description": "Potente motore da 1500 watt; Energia di impatto 10,5 J, per pesanti applicazioni di scalpellatura; La modalità di percussione morbida permette di ridurre l'energia battente per migliori risultati in materiali teneri; Il sistema di antivibrazione AVS permette di ridurre significativamente le vibrazioni, per un maggior comfort di utilizzo; Avviamento morbido per un'ottimale controllo della foratura; 'Luce service' indica la necessità di manutenzione; Indicatore luminoso di presenza della tensione", "image_url": "/images/aeg_products/Martello_demolitore_7_kg_SDS-MAX_MH_7E.webp" }
                                        ]""",
                "question": "Genera per favore due informazioni: 1) Una descrizione commerciale tecnica per Smerigliatrice angolare Brushless 18V BEWS 18-125BL. 2) Il percorso (path) dell’immagine associata a questo prodotto. Restituisci l’output in questo formato: {{ \"description\": \"<testo descrizione>\", \"image_url\": \"<percorso/immagine.webp>\"}}",
                "reasoning": """
                                1. Ho esaminato il contesto fornito e ho trovato due prodotti: una carta abrasiva e un martello demolitore.
                                2. Nessuno di questi prodotti è una smerigliatrice angolare, quindi non posso generare una descrizione per un prodotto che non è presente nel contesto.
                                3. Non posso inventare specifiche per un prodotto inesistente, quindi la risposta per la decrizione è "Non lo so."
                                4. Di conseguenza lascio l'image url vuoto.
                            """,
                "answer": "{{ \"description\": \"Non lo so.\", \"image_url\": \"\"}}"
            }
        ]
    
    def add_example(self, question: str, answer: str, context_snapshot: str, reasoning: str):
        """Aggiunge un nuovo esempio con reasoning"""        
        new_example = {
            "context_snapshot": context_snapshot.strip(),
            "question": question.strip(),
            "answer": answer.strip(),
            "reasoning": reasoning.strip()
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
    
    def format_examples_for_prompt(self, max_examples: int = 3, store=None) -> str:
        """
        Formatta gli esempi per il prompt
        
        Args:
            max_examples: Numero massimo di esempi da includere
            store: Vector store FAISS per recuperare il contesto (opzionale)
        """
        examples = self.get_examples(max_examples)
        if not examples:
            return ""
            
        formatted_examples = []
        
        for i, example in enumerate(examples, 1):
            # Salta esempi vuoti o di default
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
        
        # Filtra esempi vuoti
        valid_examples = [ex for ex in self.examples if ex.get('question') and ex.get('answer')]
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
            similar_examples = examples_store.similarity_search(query, k=min(max_examples, len(example_docs)))
            
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
            return self.format_examples_for_prompt(max_examples, store)