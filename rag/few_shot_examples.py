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
                # Crea esempi di default per il dominio edilizia
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
        """Esempi di default per il dominio edilizia"""
        return [    # TODO
            {
                "context": """
                Prodotto: Cemento Portland CEM II/A-LL 42.5 R
                Marca: Italcementi
                Codice: IT-CEM-425R
                Resistenza: 42.5 MPa a 28 giorni
                Tipo: Cemento con aggiunta di calcari
                Applicazioni: Calcestruzzo strutturale, prefabbricati
                Classe di resistenza: 42.5 R (rapido indurimento)
                Conforme a: EN 197-1
                """,
                "question": "Genera una descrizione di prodotto commerciale per AEG SMERIGLIATRICE D.115 800W. Nella descrizione non includere il nome del prodotto come prima cosa.",
                "answer": "Il cemento IT-CEM-425R è un cemento Portland di tipo CEM II/A-LL 42.5 R prodotto da Italcementi. Si tratta di un cemento con aggiunta di calcari che raggiunge una resistenza caratteristica di 42.5 MPa a 28 giorni. La sigla 'R' indica un rapido indurimento, rendendolo particolarmente adatto per applicazioni che richiedono sviluppo precoce della resistenza. È conforme alla norma EN 197-1 e viene utilizzato principalmente per calcestruzzo strutturale e nella produzione di prefabbricati."
            },
            {
                "context": """
                Prodotto: Trapano a percussione
                Marca: Bosch
                Modello: GSB 13 RE
                Potenza: 600W
                Mandrino: 13mm
                Velocità: 0-2800 giri/min
                Coppia massima: 34 Nm
                Peso: 1.8 kg
                """,
                "question": "Descrivimi il trapano GSB 13 RE",
                "answer": "Il GSB 13 RE è un trapano a percussione professionale prodotto da Bosch. Ha una potenza di 600W e un mandrino da 13mm, con velocità variabile fino a 2800 giri/min. La coppia massima è di 34 Nm, mentre il peso contenuto di 1.8 kg lo rende maneggevole per lavori prolungati. È adatto per foratura in muratura, legno e metallo grazie alla funzione percussione."
            },
            {
                "context": """
                Prodotto: Isolante termico
                Tipo: Pannelli in polistirene espanso EPS
                Marca: Stiferite
                Spessore: 50mm, 80mm, 100mm
                Lambda: 0.036 W/mK
                Resistenza termica R: da 1.39 a 2.78 m²K/W
                Classificazione fuoco: Euroclass E
                """,
                "question": "Che tipo di isolante è disponibile?",
                "answer": "Esistono più possibili corrispondenze per isolante termico."
            },
            {
                "context": """
                Vari prodotti di diverse marche per intonaci e malte...
                """,
                "question": "Descrivimi l'intonaco",
                "answer": "Esistono più possibili corrispondenze per intonaco."
            }
        ]
    
    def add_example(self, context: str, question: str, answer: str):
        """Aggiunge un nuovo esempio"""
        new_example = {
            "context": context.strip(),
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
    
    def format_examples_for_prompt(self, max_examples: int = 3) -> str:
        """Formatta gli esempi per il prompt"""
        examples = self.get_examples(max_examples)
        formatted_examples = []
        
        for i, example in enumerate(examples, 1):
            formatted_example = f"""
                Esempio {i}:
                Contesto: {example['context']}
                Domanda: {example['question']}
                Risposta: {example['answer']}
                """
            formatted_examples.append(formatted_example)
        
        return "\n".join(formatted_examples)