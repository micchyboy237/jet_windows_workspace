import os
import json
import argparse
from llama_cpp import Llama

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate LLM entity familiarity using a GGUF model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=r"C:\Users\druiv\.cache\llama.cpp\nsfw\Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        help="Path to the GGUF model file."
    )
    parser.add_argument(
        "--entities",
        type=str,
        default=None,
        help="JSON string of entities or path to a JSON file containing the entity list. If None, uses default sample."
    )
    parser.add_argument(
        "--n_ctx",
        type=int,
        default=2048,
        help="Context window size for the model."
    )
    parser.add_argument(
        "--n_gpu_layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU. -1 for all, 0 for CPU only."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature. Lower values are more deterministic."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=150,
        help="Maximum number of tokens to generate per entity."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save results as JSON. If None, prints to console."
    )
    parser.add_argument(
        "--use_json_mode",
        action="store_true",
        default=True,
        help="Use JSON schema response format to enforce structured output and suppress thinking."
    )
    return parser.parse_args()

def load_entities(entities_arg):
    """Loads entities from a JSON string, file, or returns default sample."""
    # Note: 'category' is now optional in the input list
    default_entities = [
        {"entity": "Python"},
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
        return json.loads(entities_arg)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON string for --entities")

def run_evaluation(args):
    print(f"Loading model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at: {args.model_path}")
    
    entities = load_entities(args.entities)
    print(f"Testing {len(entities)} entities...")
    
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

    # Define the JSON Schema for the response
    # The model will now generate the category itself
    json_schema = {
        "type": "object",
        "properties": {
            "identified_category": {
                "type": "string",
                "description": "The general category or type of the entity (e.g., City, Person, Technology)."
            },
            "is_familiar": {
                "type": "boolean",
                "description": "True if the entity is well-known globally, False otherwise."
            },
            "explanation": {
                "type": "string",
                "description": "A brief explanation supporting the familiarity assessment."
            }
        },
        "required": ["identified_category", "is_familiar", "explanation"]
    }

    results = []
    print("\n--- Starting Evaluation ---")
    for item in entities:
        entity_name = item.get("entity", "Unknown")
        # Use provided category if available, otherwise let the model decide
        input_category = item.get("category", None)
        
        if input_category:
            prompt_content = f"Identify the category of '{entity_name}' and determine if it is a well-known {input_category}. Answer in JSON."
        else:
            prompt_content = f"Identify the category of '{entity_name}' and determine if it is a well-known entity in that category. Answer in JSON."

        messages = [
            {"role": "system", "content": "You are an expert entity recognition assistant. Identify the entity's category and assess its familiarity. Answer strictly in the requested JSON format."},
            {"role": "user", "content": prompt_content}
        ]
        
        try:
            completion_kwargs = {
                "messages": messages,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
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
                identified_cat = response_data.get("identified_category", "Unknown")
                is_familiar = response_data.get("is_familiar", False)
                explanation = response_data.get("explanation", "No explanation provided.")
                response_text = f"[{identified_cat}] Yes: {explanation}" if is_familiar else f"[{identified_cat}] No: {explanation}"
            except json.JSONDecodeError:
                clean_text = raw_content.replace("<think>", "").replace("</think>", "").strip()
                is_familiar = False
                identified_cat = "Parse Error"
                response_text = f"JSON Error: {clean_text[:50]}"
                
        except Exception as e:
            response_text = f"Error: {str(e)}"
            is_familiar = False
            identified_cat = "Error"
        
        result_entry = {
            "entity": entity_name,
            "input_category": input_category,
            "identified_category": identified_cat,
            "model_response": response_text,
            "is_familiar": is_familiar
        }
        results.append(result_entry)
        
        status = "✅ Familiar" if is_familiar else "❌ Not Familiar"
        print(f"Entity: {entity_name:<20} | {status} | Cat: {identified_cat:<15} | Resp: {response_text[:50]}...")

    total = len(results)
    familiar_count = sum(1 for r in results if r['is_familiar'])
    accuracy = familiar_count / total if total > 0 else 0
    
    print("\n--- Evaluation Summary ---")
    print(f"Total Entities Tested: {total}")
    print(f"Entities Recognized: {familiar_count}")
    print(f"Familiarity Accuracy: {accuracy:.2%}")
    
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump({"summary": {"total": total, "familiar": familiar_count, "accuracy": accuracy}, "results": results}, f, indent=2)
        print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    args = parse_arguments()
    run_evaluation(args)