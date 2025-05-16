from langchain.chains import RetrievalQA
from .config import settings

# import LLM wrappers
from langchain_openai import OpenAI, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(provider: str, model_name: str):
    """
    Ritorna un oggetto LLM LangChain in base al provider scelto.
    provider: 'openai', 'gemini' o 'llama'
    """
    if provider == 'openai':
        return OpenAI(model_name=model_name or "gpt-3.5-turbo", openai_api_key=settings.OPENAI_API_KEY)
    elif provider == 'gemini':
        return ChatGoogleGenerativeAI(
            model=model_name or "models/gemini-1.5-flash-latest",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.9,
        )
    elif provider == 'llama':
        # LlamaCloud usa un'interfaccia compatibile con OpenAI
        return ChatOpenAI(
            model=model_name or "meta-llama/Llama-2-70b-chat-hf",  # Specifica il modello Llama che vuoi usare
            openai_api_key=settings.LLAMA_API_KEY,
            openai_api_base="https://api.cloud.llamaindex.ai/v1",  # Endpoint di LlamaCloud
            temperature=0.7,
            max_tokens=512
        )
    else:
        raise ValueError(f"Provider LLM non valido: {provider}")


def build_rag_chain(store, provider: str = 'openai', model_name: str = 'gpt-3.5-turbo'):
    """
    Costruisce un RetrievalQA chain ottimizzato per il dominio dell'edilizia
    """
    # MMR retriever per migliore diversità nei risultati
    retriever = store.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance per diversità
        search_kwargs={
            "k": 8,         # Recupera più documenti
            "fetch_k": 25,  # Considera più candidati
            "lambda_mult": 0.9  # Bilancia rilevanza e diversità
        }
    )
    
    llm = get_llm(provider, model_name)
    
    # Usa un prompt personalizzato specifico per edilizia
    from langchain.prompts import PromptTemplate
    
    template = """Sei un assistente esperto nel settore dell'edilizia. 
    Utilizza le seguenti informazioni per rispondere alla domanda dell'utente.
    
    Considera sempre il contesto italiano nelle tue risposte.
    Se la domanda riguarda normative, assicurati di specificare se si tratta di normative nazionali o locali.
    Se non conosci la risposta, dì semplicemente che non lo sai, non inventare ed in questo caso inizia la rispostacon "Non lo so".
    Non utilizzare grassetto o corsivo o altre formattazioni tipiche di markdown.
    
    Contesto:
    {context}
    
    Domanda: {question}
    
    Risposta dettagliata:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Per documenti di edilizia, "stuff" funziona bene con i chunks corretti
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
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

def supported_llama_models():
    """
    Restituisce i modelli supportati da LlamaCloud.
    """
    # Per LlamaCloud, restituisci un elenco fisso dei modelli noti
    return [
        {"id": "meta-llama/Llama-2-7b-chat-hf"},
        {"id": "meta-llama/Llama-2-13b-chat-hf"},
        {"id": "meta-llama/Llama-2-70b-chat-hf"}
    ]