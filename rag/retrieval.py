from langchain.prompts import PromptTemplate
from .config import settings
from .custom_retrieval_qa import CustomRetrievalQA, get_prompt_template
from typing import Any, Optional, Dict

# import LLM wrappers
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


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
            temperature=0,
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
            temperature=0,
            max_tokens=512
        )
    else:
        raise ValueError(f"Provider LLM non valido: {provider}")
    
def build_rag_chain_wrapper(store, provider: str = 'openai', model_name: str = 'gpt-4o-mini',
                    use_few_shot: bool = True, max_examples: int = 3,
                    is_bulk_upload: bool = True,
                    regenerateName: bool = None, generateDescription: bool = None,
                    search_kwargs: Optional[Dict[str, Any]] = None):
    """
    Wrapper per decidere quale chain costruire in base alle opzioni.
    
    I parametri `regenerateName` e `generateDescription` sono utilizzati per decidere 
    se generare il nome e la descrizione del prodotto in caso di bulk upload.
    """
    
    if is_bulk_upload:
        return build_rag_chain(
            store, provider, model_name, use_few_shot, max_examples, 
            regenerateName, generateDescription, search_kwargs=search_kwargs
        )
    else:
        return build_rag_chain(
            store, provider, model_name, use_few_shot, max_examples,
            regenerateName=None, generateDescription=None, search_kwargs=search_kwargs
        )
        
    
def build_rag_chain(store, provider: str = 'openai', model_name: str = 'gpt-4o-mini', 
                                        use_few_shot: bool = True, max_examples: int = 3,
                                        regenerateName: bool = None, generateDescription: bool = None,
                                        search_kwargs: Optional[Dict[str, Any]] = None):
    """
    Costruisce un RetrievalQA chain ottimizzato con few-shot examples dinamici,
    gestione token e supportando filtri dinamici tramite search_kwargs.
    """
    final_search_kwargs = search_kwargs if search_kwargs is not None else {"k": 5}
    
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs=final_search_kwargs
    )
    
    llm = get_llm(provider, model_name)
    
    template = get_prompt_template(regenerateName, generateDescription)

    initial_template = template.replace("{few_shot_section}", "")

    prompt = PromptTemplate(
        template=initial_template,
        input_variables=["context", "question"]
    )
    
    if regenerateName is None and generateDescription is None:
        # Caso di rigenerazione del singolo prodotto
        return CustomRetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
            use_few_shot=use_few_shot,
            max_examples=max_examples,
            provider=provider,
            model_name=model_name,
            regenerateName=None,
            generateDescription=None
        )
    else:
        # Caso di bulk upload
        return CustomRetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
            use_few_shot=use_few_shot,
            max_examples=max_examples,
            provider=provider,
            model_name=model_name,
            regenerateName=regenerateName,
            generateDescription=generateDescription
        )
    
    

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