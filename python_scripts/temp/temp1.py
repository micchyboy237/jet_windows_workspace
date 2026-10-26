import os
import json
import argparse
from llama_cpp import Llama

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate LLM entity familiarity using a GGUF model.")
    parser.add_argument(
        "-m", "--model_path",
        type=str,
        default=r"C:\Users\druiv\.cache\llama.cpp\nsfw\Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        help="Path to the GGUF model file."
    )
    parser.add_argument(
        "-c", "--n_ctx",
        type=int,
        default=2048,
        help="Context window size for the model."
    )
    parser.add_argument(
        "-g", "--n_gpu_layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU. -1 for all, 0 for CPU only."
    )
    parser.add_argument(
        "-e", "--entities",
        type=str,
        default=None,
        help="JSON string, path to JSON file, or comma-separated list of entities (e.g., 'Python,Paris,Elon Musk')."
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Official recommendation: 1.0 for general tasks."
    )
    parser.add_argument(
        "-p", "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter. Official recommendation: 0.95."
    )
    parser.add_argument(
        "-k", "--top_k",
        type=int,
        default=20,
        help="Top-k sampling parameter. Official recommendation: 20."
    )
    parser.add_argument(
        "--min_p",
        type=float,
        default=0.0,
        help="Min-p sampling parameter."
    )
    parser.add_argument(
        "--pp", "--presence_penalty",
        type=float,
        default=1.5,
        dest="presence_penalty",
        help="Presence penalty. Official recommendation: 1.5."
    )
    parser.add_argument(
        "--rp", "--repeat_penalty",
        type=float,
        default=1.0,
        dest="repeat_penalty",
        help="Repeat penalty. Official recommendation: 1.0."
    )
    parser.add_argument(
        "-n", "--max_tokens",
        type=int,
        default=500,
        help="Maximum number of tokens to generate per entity."
    )
    parser.add_argument(
        "-s", "--stream",
        action="store_true",
        default=True,
        help="Stream the output token by token."
    )
    parser.add_argument(
        "-j", "--use_json_mode",
        action="store_true",
        default=True,
        help="Use JSON schema response format to enforce structured output."
    )
    parser.add_argument(
        "-o", "--output_file",
        type=str,
        default=None,
        help="Path to save results as JSON. If None, prints to console."
    )
    return parser.parse_args()

def load_entities(entities_arg):
    """Loads entities from a JSON string, file, comma-separated string, or returns default sample."""
    default_entities = [
        {"entity": "Python"},
        {"entity": "RAG"},
        {"entity": "ReAcT Agentic Loop"},
        {"entity": "Paris"},
        {"entity": "Elon Musk"},
        {"entity": "Quantum Computing"},
        {"entity": "The Great Gatsby"}
    ]
    if entities_arg is None:
        return default_entities
    if os.path.isfile(entities_arg):
        with open(entities_arg, 'r') as f:
            return json.load(f)
    try:
        parsed_json = json.loads(entities_arg)
        if isinstance(parsed_json, list):
            if all(isinstance(item, str) for item in parsed_json):
                return [{"entity": item} for item in parsed_json]
            return parsed_json
    except json.JSONDecodeError:
        pass
    if "," in entities_arg:
        items = [item.strip() for item in entities_arg.split(",") if item.strip()]
        return [{"entity": item} for item in items]
    return [{"entity": entities_arg}]

def stream_response(llm, messages, args, json_schema=None):
    """Stream the response token by token and return the full content."""
    completion_kwargs = {
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repeat_penalty": args.repeat_penalty,
        "stop": ["<|im_end|>"],
        "stream": True
    }
    if args.use_json_mode and json_schema:
        completion_kwargs["response_format"] = {
            "type": "json_object",
            "schema": json_schema
        }
    full_content = ""
    print("  Response: ", end="", flush=True)
    try:
        for chunk in llm.create_chat_completion(**completion_kwargs):
            delta = chunk['choices'][0]['delta']
            content = delta.get('content', '')
            if content:
                print(content, end="", flush=True)
                full_content += content
    except Exception as e:
        print(f"\n  Streaming error: {e}", flush=True)
    print()
    return full_content

def run_evaluation(args):
    print(f"Loading model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at: {args.model_path}")
    
    entities = load_entities(args.entities)
    print(f"Testing {len(entities)} entities...")
    print(f"\nGeneration Settings:")
    print(f"  Temp: {args.temperature} | Top-p: {args.top_p} | Top-k: {args.top_k}")
    print(f"  PresPen: {args.presence_penalty} | RepPen: {args.repeat_penalty}")
    print(f"  MaxTokens: {args.max_tokens} | Stream: {args.stream}\n")
    
    llama_kwargs = {
        "model_path": args.model_path,
        "n_ctx": args.n_ctx,
        "n_gpu_layers": args.n_gpu_layers,
        "verbose": False,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    try:
        llm = Llama(**llama_kwargs)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    json_schema = {
        "type": "object",
        "properties": {
            "entity_name": {
                "type": "string",
                "description": "The name of the entity being evaluated."
            },
            "meanings": {
                "type": "array",
                "description": "An array of different meanings or interpretations of the entity (homonyms).",
                "items": {
                    "type": "object",
                    "properties": {
                        "identified_category": {
                            "type": "string",
                            "description": "The general category or type of this specific meaning (e.g., City, Person, Technology, Animal)."
                        },
                        "score": {
                            "type": "number",
                            "description": "A familiarity score from 0.0 to 1.0 based on how extensively this term appeared in your training data. Score 1.0 if this term was very common and well-represented in training data, 0.0 if it was rare or absent. This reflects your model's exposure to this concept during training."
                        },
                        "explanation": {
                            "type": "string",
                            "description": "A brief explanation supporting the familiarity assessment for this specific meaning."
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional context or domain where this meaning is commonly used."
                        }
                    },
                    "required": ["identified_category", "score", "explanation"]
                },
                "minItems": 1,
                "maxItems": 5
            }
        },
        "required": ["entity_name", "meanings"]
    }

    system_prompt = (
        "You are an advanced semantic analysis engine. Your goal is to deconstruct entities into their distinct semantic meanings (homonyms). "
        "For each input entity:\n"
        "1. Identify all valid, distinct interpretations (e.g., 'Apple' as a fruit vs. 'Apple' as a tech company).\n"
        "2. Categorize each interpretation precisely.\n"
        "3. Assign a familiarity score (0.0 to 1.0) for each interpretation based on how extensively this term appeared in your training data. A score of 1.0 means this term was very common and well-represented in your training corpus, while 0.0 means it was rare or virtually absent. This score should reflect your model's exposure to this concept during training.\n"
        "4. Provide a concise explanation for your assessment.\n"
        "CRITICAL RULES:\n"
        "- Distinguish between the entity itself and related tools/concepts (e.g., 'Java' is a language/island, not 'IntelliJ').\n"
        "- Return ONLY valid JSON matching the provided schema. Do not include markdown formatting or conversational text.\n"
        "- If an entity has no common homonyms, provide the primary meaning."
    )

    results = []
    print("\n--- Starting Evaluation ---")
    for item in entities:
        entity_name = item.get("entity", "Unknown")
        input_category = item.get("category", None)
        if input_category:
            prompt_content = f"Analyze the entity '{entity_name}'. Specifically, check for meanings related to: {input_category}. Return all findings in the requested JSON format."
        else:
            prompt_content = f"Analyze the entity '{entity_name}' for all possible semantic meanings and homonyms. Return all findings in the requested JSON format."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_content}
        ]
        
        try:
            if args.stream:
                raw_content = stream_response(llm, messages, args, json_schema if args.use_json_mode else None)
            else:
                completion_kwargs = {
                    "messages": messages,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "min_p": args.min_p,
                    "presence_penalty": args.presence_penalty,
                    "repeat_penalty": args.repeat_penalty,
                    "stop": ["<|im_end|>"]
                }
                if args.use_json_mode:
                    completion_kwargs["response_format"] = {
                        "type": "json_object",
                        "schema": json_schema
                    }
                output = llm.create_chat_completion(**completion_kwargs)
                raw_content = output['choices'][0]['message']['content']
            
            try:
                response_data = json.loads(raw_content)
                entity_name_resp = response_data.get("entity_name", entity_name)
                meanings = response_data.get("meanings", [])
                
                meaning_results = []
                for idx, meaning in enumerate(meanings):
                    identified_cat = meaning.get("identified_category", "Unknown")
                    score = meaning.get("score", 0.0)
                    explanation = meaning.get("explanation", "No explanation provided.")
                    context = meaning.get("context", "")
                    
                    if score >= 0.7:
                        status = f"✅ High ({score:.2f})"
                    elif score >= 0.4:
                        status = f"⚠️ Medium ({score:.2f})"
                    else:
                        status = f"❌ Low ({score:.2f})"
                    
                    context_str = f" [{context}]" if context else ""
                    print(f"    Meaning {idx+1}: [{identified_cat}]{context_str} | {status}")
                    
                    meaning_results.append({
                        "identified_category": identified_cat,
                        "score": score,
                        "explanation": explanation,
                        "context": context
                    })
                
                if meanings:
                    primary_meaning = meanings[0]
                    primary_score = primary_meaning.get('score', 0.0)
                    response_text = f"[{primary_meaning.get('identified_category', 'Unknown')}] Score: {primary_score:.2f}: {primary_meaning.get('explanation', '')}"
                else:
                    response_text = "No meanings identified"
                    
            except json.JSONDecodeError:
                clean_text = raw_content.replace("<think>", "").replace("</think>", "").strip()
                response_text = f"JSON Error: {clean_text[:50]}"
                meaning_results = []
        except Exception as e:
            response_text = f"Error: {str(e)}"
            meaning_results = []
        
        result_entry = {
            "entity": entity_name,
            "input_category": input_category,
            "meanings": meaning_results,
            "model_response": response_text,
            "total_meanings": len(meaning_results)
        }
        results.append(result_entry)
        total_meanings = len(meaning_results)
        high_scores = sum(1 for m in meaning_results if m.get('score', 0) >= 0.7)
        print(f"Entity: {entity_name:<20} | Meanings: {total_meanings} | High Score: {high_scores}/{total_meanings}\n")
    
    total = len(results)
    total_meanings_all = sum(r['total_meanings'] for r in results)
    high_score_meanings_all = sum(sum(1 for m in r['meanings'] if m.get('score', 0) >= 0.7) for r in results)
    avg_score = sum(m.get('score', 0) for r in results for m in r['meanings']) / total_meanings_all if total_meanings_all > 0 else 0
    
    print("--- Evaluation Summary ---")
    print(f"Total Entities Tested: {total}")
    print(f"Total Meanings Identified: {total_meanings_all}")
    print(f"High Score Meanings (>=0.7): {high_score_meanings_all}")
    print(f"Average Familiarity Score: {avg_score:.2f}")
    
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump({
                "summary": {
                    "total_entities": total,
                    "total_meanings": total_meanings_all,
                    "high_score_meanings": high_score_meanings_all,
                    "average_score": avg_score
                },
                "results": results
            }, f, indent=2)
        print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    args = parse_arguments()
    run_evaluation(args)