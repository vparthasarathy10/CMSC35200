import json
import argparse
import requests
import time
import openai
import re

def load_questions(filepath):
    with open(filepath, 'r') as file:
        questions = json.load(file)
    return questions

def get_llama_response(question, model_info):
    try:
        openai.api_key = "cmsc-35360"
        openai.api_base = f"http://{model_info['api_base']}:{model_info['port']}/v1"

        chat_response = openai.ChatCompletion.create(
            model=model_info['model_name'],
            messages=[
                {
                    "role": "user", 
                    "content": f"Answer the following question. Only provide the numerical answer with no explanation, no elaboration, and no punctuation. Do not use phrases like 'x ='. Just give the numbers: {question}"
                }
            ],
            temperature=0.0,
            max_tokens=2056
        )
        return chat_response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error with {model_info['model_name']}: {e}")
        return ""

def evaluate_models(questions, models):
    results = {}
    for model in models:
        model_name = model['name']
        results[model_name] = {'total': len(questions), 'responses': []}
        print(f"Evaluating Model: {model_name}")
        
        for q in questions:
            question = q['question']
            correct_answer = q['answer']
            
            print(f"Question: {question}")
            print(f"Correct Answer: {correct_answer}")
            
            answer = get_llama_response(question, model_info=model)
            print(f"Model Answer: {answer}\n")
            
            results[model_name]['responses'].append({
                'question': question,
                'correct_answer': correct_answer,
                'model_answer': answer
            })
            
            # Sleep to avoid overwhelming the API
            time.sleep(1)
    
    return results

def display_results(results):
    print("=== Results (Manual Evaluation Needed) ===\n")
    for model, data in results.items():
        print(f"Model: {model}")
        for resp in data['responses']:
            print(f"Question: {resp['question']}")
            print(f"Correct Answer: {resp['correct_answer']}")
            print(f"Model Answer: {resp['model_answer']}\n")
        print("----------------------------------------------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Llama Models on a Set of Questions.")
    parser.add_argument("--questions", type=str, default="questions.json", help="Path to the questions JSON file.")
    args = parser.parse_args()

    # Load questions
    questions = load_questions(args.questions)

    # Define models to evaluate
    models = [
        {
            "name": "Llama 3.1 8B",
            "api_base": "103.101.203.226",
            "port": 80,
            "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct"
        },
        {
            "name": "Llama 3.1 70B",
            "api_base": "195.88.24.64",
            "port": 80,
            "model_name": "meta-llama/Meta-Llama-3.1-70B-Instruct"
        },
        {
            "name": "Llama 3.1 405B",
            "api_base": "66.55.67.65",
            "port": 80,
            "model_name": "llama31-405b-fp8"
        }
    ]

    # Evaluate models
    results = evaluate_models(questions, models)

    # Display results
    display_results(results)

    # Save results to a JSON file
    with open('evaluation_results.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)
    print("Results saved to evaluation_results.json")

if __name__ == "__main__":
    main()
