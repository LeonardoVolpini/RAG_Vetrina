from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from .config import settings
from .few_shot_examples import FewShotExampleManager
from typing import Any, Optional, List
from pydantic import PrivateAttr
import tiktoken

# import LLM wrappers
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# import prompts template
from .prompts.description_image_prompt import get_description_and_image_template
from .prompts.name_description_image_prompt import get_name_description_image_template
from .prompts.image_prompt import get_image_template
from .prompts.name_image_prompt import get_name_and_image_template

def get_prompt_template(regenerateName: bool = False, generateDescription: bool = True) -> str:
    """
    Restituisce il template del prompt in base alle opzioni di generazione
    """
    if (regenerateName and generateDescription):
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


def get_token_limits(model_name: str) -> dict:
    """
    Restituisce i limiti di token per diversi modelli
    """
    limits = {
        # OpenAI models
        "gpt-3.5-turbo": {"total": 8192, "max_context": 6000},
        "gpt-3.5-turbo-16k": {"total": 16384, "max_context": 14000},
        "gpt-4": {"total": 8192, "max_context": 6000},
        "gpt-4-32k": {"total": 32768, "max_context": 28000},
        "gpt-4-turbo": {"total": 128000, "max_context": 120000},
        "gpt-4o": {"total": 128000, "max_context": 120000},
        "gpt-4o-mini": {"total": 128000, "max_context": 120000},
        
        # Gemini models (approssimativi)
        "models/gemini-1.5-flash-latest": {"total": 1000000, "max_context": 950000},
        "models/gemini-1.5-pro-latest": {"total": 2000000, "max_context": 1900000},
        "models/gemini-2.5-flash-preview-05-20": {"total": 1000000, "max_context": 950000},
        
        # Llama models (approssimativi)
        "meta-llama/Llama-2-7b-chat-hf": {"total": 4096, "max_context": 3000},
        "meta-llama/Llama-2-13b-chat-hf": {"total": 4096, "max_context": 3000},
        "meta-llama/Llama-2-70b-chat-hf": {"total": 4096, "max_context": 3000},
        "meta-llama/Meta-Llama-3-8B-Instruct": {"total": 8192, "max_context": 6000},
        "meta-llama/Meta-Llama-3-70B-Instruct": {"total": 8192, "max_context": 6000},
    }
    
    # Default per modelli non riconosciuti
    return limits.get(model_name, {"total": 4096, "max_context": 3000})


def get_encoding_for_model(provider: str, model_name: str):
    """
    Restituisce l'encoding tiktoken appropriato per il modello
    """
    if provider == 'openai':
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback per modelli non riconosciuti
            return tiktoken.get_encoding("cl100k_base")
    
    elif provider == 'gemini':
        # Gemini usa un tokenizer diverso, ma per approssimazione usiamo cl100k_base
        return tiktoken.get_encoding("cl100k_base")
    
    elif provider == 'llama':
        # Llama usa un tokenizer diverso, ma per approssimazione usiamo cl100k_base
        return tiktoken.get_encoding("cl100k_base")
    
    else:
        # Default
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, encoding) -> int:
    """
    Conta i token in un testo usando l'encoding specificato
    """
    if not text or text.strip() == "":
        return 0
        
    try:
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Errore nel conteggio token: {e}")
        # Fallback più accurato: ~3.5 caratteri per token (più preciso di 4)
        return max(1, len(text.strip()) // 4)


def truncate_text_by_tokens(text: str, max_tokens: int, encoding, preserve_structure: bool = True) -> str:
    """
    Tronca il testo per rispettare il limite di token
    """
    if not text or max_tokens <= 0:
        return ""
    
    try:
        tokens = encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        # Tronca i token
        truncated_tokens = tokens[:max_tokens]
        truncated_text = encoding.decode(truncated_tokens)
        
        # Se preserve_structure è True, prova a mantenere frasi complete
        if preserve_structure and max_tokens > 50:  # Solo se abbiamo abbastanza token
            # Trova l'ultimo punto, a capo o fine frase
            last_sentence_end = max(
                truncated_text.rfind('.'),
                truncated_text.rfind('\n'),
                truncated_text.rfind('!'),
                truncated_text.rfind('?')
            )
            
            # Se troviamo una fine frase ragionevole (non troppo vicina all'inizio)
            if last_sentence_end > len(truncated_text) * 0.7:  # Almeno 70% del testo
                truncated_text = truncated_text[:last_sentence_end + 1]
        
        return truncated_text
    
    except Exception as e:
        print(f"Errore nel troncamento: {e}")
        # Fallback: tronca per caratteri
        estimated_chars = max_tokens * 4
        return text[:estimated_chars] if len(text) > estimated_chars else text
    

def optimize_full_prompt_for_model(docs: List[Any], question: str, few_shot_section: str, 
                                   template: str, provider: str, model_name: str) -> tuple[str, str, dict]:
    """
    Ottimizza l'intero prompt mantenendo il tuo template base originale
    """
    # Ottieni limiti e encoding
    limits = get_token_limits(model_name)
    max_total_tokens = limits["total"]
    encoding = get_encoding_for_model(provider, model_name)
    
    # Prepara il contesto completo dai documenti
    full_context = "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)
    
    # Calcola i token di ogni componente
    question_tokens = count_tokens(question, encoding)
    context_tokens = count_tokens(full_context, encoding)
    few_shot_tokens = count_tokens(few_shot_section, encoding) if few_shot_section else 0
    
    # Template senza le parti dinamiche per calcolare i token fissi
    template_fixed = template.replace("{few_shot_section}", "").replace("{context}", "PLACEHOLDER_CONTEXT").replace("{question}", "PLACEHOLDER_QUESTION")
    template_tokens = count_tokens(template_fixed, encoding)
    
    # Token disponibili (lasciamo margine per la risposta del modello)
    response_margin = min(512, max_total_tokens // 4)  # 25% del limite o 512, quello che è minore
    available_tokens = max_total_tokens - template_tokens - question_tokens - response_margin
    
    #print(f"\n📊 ANALISI COMPLETA TOKEN ({model_name}):")
    #print(f"   Limite totale: {max_total_tokens} token")
    #print(f"   Template fisso: {template_tokens} token")
    #print(f"   Question: {question_tokens} token")
    #print(f"   Margine risposta: {response_margin} token")
    #print(f"   Disponibili per contenuto: {available_tokens} token")
    #print(f"   ---")
    #print(f"   Few-shot originali: {few_shot_tokens} token")
    #print(f"   Contesto originale: {context_tokens} token")
    #print(f"   Totale contenuto dinamico: {few_shot_tokens + context_tokens} token")
    #print(f"   ---")
    
    # Statistiche per il return
    stats = {
        "total_limit": max_total_tokens,
        "template_tokens": template_tokens,
        "question_tokens": question_tokens,
        "response_margin": response_margin,
        "available_tokens": available_tokens,
        "original_few_shot": few_shot_tokens,
        "original_context": context_tokens
    }
    
    # Ricorsiva/iterativa: riduci i few-shot finché serve
    reduction_step = 0.05  # 5%
    min_few_shot_tokens = 10  # soglia minima di token per non azzerare tutto

    while True:
        # Se il contenuto è troppo lungo, ottimizza
        if few_shot_tokens + context_tokens > available_tokens:
            print(f"⚠️  Prompt troppo lungo ({few_shot_tokens + context_tokens} > {available_tokens}), riduco few-shot del 5%...")
            if few_shot_tokens > min_few_shot_tokens:
                new_tokens = int(few_shot_tokens * (1 - reduction_step))
                few_shot_section = truncate_text_by_tokens(few_shot_section, new_tokens, encoding, preserve_structure=True)
                few_shot_tokens = count_tokens(few_shot_section, encoding)
            else:
                # Se i few-shot sono già minimi, riduci il contesto
                if context_tokens > 0:
                    context_tokens_new = int(context_tokens * (1 - reduction_step))
                    full_context = truncate_text_by_tokens(full_context, context_tokens_new, encoding, preserve_structure=True)
                    context_tokens = count_tokens(full_context, encoding)
                else:
                    break  # Non si può ridurre oltre
        else:
            print(f"✅ Contenuto OK: {few_shot_tokens + context_tokens}/{available_tokens} token disponibili")
            break
    
    # Verifica finale e calcolo del prompt completo
    optimized_template = template.replace("{few_shot_section}", few_shot_section)
    final_prompt = optimized_template.format(context=full_context, question=question)
    final_tokens = count_tokens(final_prompt, encoding)
    
    # Aggiorna statistiche finali
    stats.update({
        "final_few_shot": count_tokens(few_shot_section, encoding) if few_shot_section else 0,
        "final_context": count_tokens(full_context, encoding),
        "final_total": final_tokens,
        "utilization_percent": (final_tokens / max_total_tokens) * 100
    })
    
    #print(f"\n✅ OTTIMIZZAZIONE COMPLETATA:")
    #print(f"   Few-shot finale: {stats['final_few_shot']} token")
    #print(f"   Contesto finale: {stats['final_context']} token")
    #print(f"   Prompt totale: {final_tokens} token")
    #print(f"   Utilizzo: {stats['utilization_percent']:.1f}% ({final_tokens}/{max_total_tokens})")
    
    if final_tokens > max_total_tokens:
        print(f"   ❌ ERRORE: Prompt ancora troppo lungo!")
        # Ultimo tentativo di riduzione aggressiva del contesto
        emergency_context_tokens = max_total_tokens - template_tokens - question_tokens - response_margin - stats['final_few_shot']
        if emergency_context_tokens > 0:
            print(f"   🚨 Riduzione di emergenza contesto a {emergency_context_tokens} token")
            full_context = truncate_text_by_tokens(full_context, emergency_context_tokens, encoding, preserve_structure=False)
    elif final_tokens > max_total_tokens * 0.9:
        print(f"   ⚠️  WARNING: Utilizzo alto del limite token")
    else:
        print(f"   ✅ Token management OK")
    
    return full_context, few_shot_section, stats

def get_llm(provider: str, model_name: str):
    """
    Ritorna un oggetto LLM LangChain in base al provider scelto.
    provider: 'openai', 'gemini' o 'llama'
    """
    if provider == 'openai':
        return ChatOpenAI(model=model_name or "gpt-4o-mini", openai_api_key=settings.OPENAI_API_KEY)
    elif provider == 'gemini':
        return ChatGoogleGenerativeAI(
            model=model_name or "models/gemini-1.5-flash-latest",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.3,
        )
    elif provider == 'llama':        
        if not model_name:
            model_name = "meta-llama/Llama-2-70b-chat-hf"  # Default model
            
        # Log model selection
        print(f"Using LlamaCloud model: {model_name}")
        
        return ChatOpenAI(
            model=model_name,
            openai_api_key=settings.LLAMA_API_KEY,
            openai_api_base=settings.LLAMA_API_BASE,
            temperature=0.3,
            max_tokens=512
        )
    else:
        raise ValueError(f"Provider LLM non valido: {provider}")
    
def build_rag_chain_with_improved_tokens(store, provider: str = 'openai', model_name: str = 'gpt-4o-mini', 
                                        use_few_shot: bool = True, max_examples: int = 3,
                                        regenerateName: bool = False, generateDescription: bool = True):
    """
    Costruisce un RetrievalQA chain ottimizzato con few-shot examples dinamici e gestione token
    """
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,
            "score_threshold": 0.95
        }
    )
    
    llm = get_llm(provider, model_name)
    
    # Crea una versione personalizzata del RetrievalQA che include few-shot examples dinamici
    class CustomRetrievalQA(RetrievalQA):
        _use_few_shot: bool     = PrivateAttr()
        _max_examples: int      = PrivateAttr()
        _example_manager: Any   = PrivateAttr()
        _provider: str          = PrivateAttr()
        _model_name: str        = PrivateAttr()


        def __init__(self, *args: Any, **kwargs: Any):
            # Estrai e rimuovi i custom params
            _ufs = kwargs.pop("use_few_shot", True)
            _mex = kwargs.pop("max_examples", 3)
            _prov = kwargs.pop("provider", "openai")
            _model = kwargs.pop("model_name", "gpt-4o-mini")

            # Costruttore base: tutti gli altri argomenti
            super().__init__(*args, **kwargs)

            # Ora assegna i PrivateAttr bypassando Pydantic
            object.__setattr__(self, "_use_few_shot", _ufs)
            object.__setattr__(self, "_max_examples", _mex)
            object.__setattr__(self, "_provider", _prov)
            object.__setattr__(self, "_model_name", _model)
            object.__setattr__(
                self,
                "_example_manager",
                FewShotExampleManager() if _ufs else None
            )

        # Proprietà per accedere ai tuoi PrivateAttr
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
            # -------------------------------------------------
            
            # Aggiungi few-shot examples se abilitati
            few_shot_section = ""
            if self.use_few_shot and self.example_manager:
                try:
                    few_shot_examples = self.example_manager.get_relevant_examples(
                        question, 
                        self.retriever.vectorstore,
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
                    
            template = get_prompt_template(regenerateName, generateDescription)
            
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

    template = get_prompt_template(regenerateName, generateDescription)
    initial_template = template.replace("{few_shot_section}", "")

    prompt = PromptTemplate(
        template=initial_template,
        input_variables=["context", "question"]
    )
    
    return CustomRetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        use_few_shot=use_few_shot,
        max_examples=max_examples,
        provider=provider,
        model_name=model_name
    )


def build_rag_chain(store, provider: str = 'openai', model_name: str = 'gpt-4o-mini',
                    use_few_shot: bool = True, max_examples: int = 3,
                    regenerateName: bool = False, generateDescription: bool = True):
    """
    Wrapper per backward compatibility con ottimizzazione tiktoken migliorata
    """
    return build_rag_chain_with_improved_tokens(store, provider, model_name, use_few_shot, max_examples, regenerateName, generateDescription)

def supported_gemini_models():
    """
    Restituisce i modelli supportati da Gemini.
    """
    from .config import settings
    import google.generativeai as genai
    genai.configure(api_key=settings.GEMINI_API_KEY)
    models = genai.list_models()
    for m in models:
        if "generateContent" in m.supported_generation_methods:
            print(f"Model: {m.name}, metodi: {m.supported_generation_methods}")
    return models