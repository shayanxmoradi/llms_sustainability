import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import argparse
import os




def setup_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True
    )
    return pipeline(
        "text-generation",
        model=model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        tokenizer=tokenizer,
        do_sample=True,              # Random sampling
        temperature=0.9,             # Add more randomness
        top_p=0.95
    )

def get_scores(pipe, statement):
    messages = [
        {"role": "system", "content": "Respond to each of the following statements with one of the following string values on a scale from strong disagreement to strong agreement: 'Strongly Disagree', 'Disagree', 'Somewhat Disagree',  'Neutral', ''Somewhat Agree', 'Agree', or 'Strongly Agree'. Do not say anything else. Do not repeat the same response across all runs. Each run should be treated independently."},
        {"role": "user", "content": statement}
    ]

    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipe(
        messages,
        max_new_tokens=50,
        eos_token_id=terminators,
        do_sample=True
    )

    for message in outputs[0]["generated_text"]:
        if message.get('role') == 'assistant':
            return message.get('content', 'No content')
    
def process_surveys(pipe, statements, num_surveys):
    all_responses = []

    for survey_number in range(1, num_surveys + 1):
        survey_responses = {}
        for statement in statements:
            response = get_scores(pipe, statement)
            survey_responses[statement] = response
            print(f"Survey {survey_number}, Processed statement: {statement[:30]}... with response: {response}")

        all_responses.append(survey_responses)

    return pd.DataFrame(all_responses)

def main(model_id, num_surveys, output_file):
    statements = [
    'It is important to develop a mutual understanding of responsibilities regarding environmental performance with our suppliers',
    'It is important to work together to reduce environmental impact of our activities with our suppliers',
    'It is important to conduct joint planning to anticipate and resolve environmental-related problems with our suppliers',
    'It is important to make joint decisions about ways to reduce overall environmental impact of our products with our suppliers',
    'It is important to develop a mutual understanding of responsibilities regarding environmental performance with our customers',
    'It is important to work together to reduce environmental impact of our activities with our customers',
    'It is important to conduct joint planning to anticipate and resolve environmental-related problems with our customers',
    'It is important to make joint decisions about ways to reduce overall environmental impact of our products with our customers'
   ]

    pipe = setup_model(model_id)
    df = process_surveys(pipe, statements, num_surveys)
    df.to_excel(output_file, index_label='Survey Number')
    print(f"Completed all surveys and saved to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run survey processing")
    parser.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.3", help="Model ID to use")
    parser.add_argument("--num_surveys", type=int, default=100, help="Number of surveys to process")
    parser.add_argument("--output_file", default="GSC_MISTRAL.xlsx", help="Output file name")

    args = parser.parse_args()

    main(args.model_id, args.num_surveys, args.output_file)
