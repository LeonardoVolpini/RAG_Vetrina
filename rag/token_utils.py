import tiktoken
from typing import Any, List


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
    Ottimizza l'intero prompt mantenendo il template base originale
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