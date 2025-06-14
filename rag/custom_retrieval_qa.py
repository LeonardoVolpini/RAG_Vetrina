from langchain.chains import RetrievalQA
from pydantic import PrivateAttr
from typing import Any, Optional
from .few_shot_examples import FewShotExampleManager
from .token_utils import optimize_full_prompt_for_model

# import prompts template
from .prompts.description_image_prompt import get_description_and_image_template
from .prompts.name_description_image_prompt import get_name_description_image_template
from .prompts.image_prompt import get_image_template
from .prompts.name_image_prompt import get_name_and_image_template
from .prompts.single_product import get_single_product_template


def get_prompt_template(regenerateName: bool = None, generateDescription: bool = None) -> str:
    """
    Restituisce il template del prompt in base alle opzioni di generazione
    """
    if regenerateName is None and generateDescription is None:
        # Caso di rigenerazione del singolo prodotto
        print("🔄 Generazione descrizione e immagine per singolo prodotto")
        template = get_single_product_template()
    elif (regenerateName and generateDescription):
        # Usa il template per nome, descrizione e immagine
        print("🔄 Generazione nome, descrizione e immagine")
        template = get_name_description_image_template()
    elif regenerateName and not generateDescription:
        # Usa il template per nome e immagine
        print("🔄 Generazione nome e immagine")
        template = get_name_and_image_template()
    elif not regenerateName and generateDescription:
        # Usa il template per descrizione e immagine
        print("🔄 Generazione descrizione e immagine")
        template = get_description_and_image_template()
    else:
        # Usa il template per solo immagine
        print("🔄 Generazione immagine")
        template = get_image_template()
    return template

class CustomRetrievalQA(RetrievalQA):
    """
    Versione personalizzata del RetrievalQA che include:
    - Few-shot examples dinamici
    - Gestione ottimizzata dei token
    - Supporto per filtri dinamici
    """
    _use_few_shot: bool = PrivateAttr()
    _max_examples: int = PrivateAttr()
    _example_manager: Any = PrivateAttr()
    _provider: str = PrivateAttr()
    _model_name: str = PrivateAttr()
    _regenerateName: bool = PrivateAttr()           # Se != None -> bulk upload
    _generateDescription: bool = PrivateAttr()      # Se != None -> bulk upload

    def __init__(self, *args: Any, **kwargs: Any):
        # Estrai e rimuovi i custom params
        _ufs = kwargs.pop("use_few_shot", True)
        _mex = kwargs.pop("max_examples", 3)
        _prov = kwargs.pop("provider", "openai")
        _model = kwargs.pop("model_name", "gpt-4o-mini")
        _name = kwargs.pop("regenerateName", None)
        _desc = kwargs.pop("generateDescription", None)

        # Costruttore base: tutti gli altri argomenti
        super().__init__(*args, **kwargs)

        # Ora assegna i PrivateAttr bypassando Pydantic
        object.__setattr__(self, "_use_few_shot", _ufs)
        object.__setattr__(self, "_max_examples", _mex)
        object.__setattr__(self, "_provider", _prov)
        object.__setattr__(self, "_model_name", _model)
        object.__setattr__(self, "_regenerateName", _name)
        object.__setattr__(self, "_generateDescription", _desc)
        object.__setattr__(
            self,
            "_example_manager",
            FewShotExampleManager() if _ufs else None
        )

    # Proprietà per accedere ai PrivateAttr
    @property
    def use_few_shot(self) -> bool:
        return self._use_few_shot

    @property
    def max_examples(self) -> int:
        return self._max_examples

    @property
    def example_manager(self) -> Optional[Any]:
        return self._example_manager
    
    @property
    def provider(self) -> str:
        return self._provider
        
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def regenerateName(self) -> bool:
        return self._regenerateName
    
    @property
    def generateDescription(self) -> bool:
        return self._generateDescription
    
    def _get_docs(self, question: str, *, run_manager=None):
        """Override per includere few-shot examples nella chiamata e ottimizzazione token"""
        docs = super()._get_docs(question, run_manager=run_manager)
        
        # ---- PER DEBUG SIMILARITA' CON SCORE E SENZA ---
        #results = self.retriever.vectorstore.similarity_search_with_score(question, k=3)
        #docs = [doc for doc, score in results]
        #print(f"Documenti selezionati {len(docs)} dal retriever (con punteggio):")
        #for i, (doc, score) in enumerate(results):
        #    print(f"[{i+1}] Similarità: {score:.4f} | {getattr(doc, 'page_content', str(doc))[:500]}")
        
        # Stampa i documenti selezionati dal retriever
        print(f"Documenti selezionati {len(docs)} dal retriever:")
        for i, doc in enumerate(docs):
            print(f"[{i+1}] {getattr(doc, 'page_content', str(doc))}")
        
        # Aggiungi few-shot examples se abilitati
        few_shot_section = ""
        if self.use_few_shot and self.example_manager:
            try:
                few_shot_examples = self.example_manager.get_relevant_examples(
                    question,
                    self.regenerateName,
                    self.generateDescription,
                    self.max_examples
                )
                
                if few_shot_examples:
                    few_shot_section = f"""
                    <few_shot_examples>
                    Ecco alcuni esempi rilevanti di come rispondere correttamente:
                    {few_shot_examples}
                    </few_shot_examples>

                    Studia attentamente questi esempi per comprendere:
                    - Come identificare prodotti specifici vs generici
                    - Quando dire "Non lo so" vs "Esistono più possibili corrispondenze"
                    - Il livello di dettaglio tecnico appropriato
                    - Il tono e lo stile delle risposte

                    """
                    
            except Exception as e:
                print(f"Errore nel recupero few-shot examples: {str(e)}")
                few_shot_section = ""
        
        template = get_prompt_template(self.regenerateName, self.generateDescription)
        
        optimized_context, optimized_few_shot, token_stats = optimize_full_prompt_for_model(
            docs, question, few_shot_section, template, self.provider, self.model_name
        )
        
        # Aggiorna il template con il contenuto ottimizzato
        updated_template = template.replace("{few_shot_section}", optimized_few_shot)
        self.combine_documents_chain.llm_chain.prompt.template = updated_template
        
        # Stampa il prompt finale ottimizzato
        #final_prompt = updated_template.format(context=optimized_context, question=question)
        #print("\n===== PROMPT FINALE OTTIMIZZATO =====\n")
        #print(final_prompt)
        #print("\n=====================================\n") 
           
        return docs