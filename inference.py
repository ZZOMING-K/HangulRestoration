import os
from tqdm import tqdm

import pandas as pd 
import torch
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import argparse 
from transformers import set_seed
from datasets import Dataset, DatasetDict
    
def generate_dict(df) : 
    
    instruction_list = [ [open('./data/instruction.txt').read()] for _ in range(len(df)) ] 
    input_list = df['input']
    output_list = df['output']
    
    dataset_dict = {'instruction' : instruction_list , 'input' : input_list , 'output' : output_list} 
    dataset = Dataset.from_dict(dataset_dict)
    
    return dataset 
    
def create_datasets(df , tokenizer):
    
    def preprocess(samples):
        
        batch = []
        PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context.\n"
            "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
            "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
            "### Instruction(명령어):{instruction}\n\n### Input(입력):{input}\n\n### Response(응답):{response}"
        ),
        "prompt_inference": (
            "Below is an instruction that describes a task, paired with an input that provides further context.\n"
            "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
            "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
            "### Instruction(명령어):{instruction}\n\n### Input(입력):{input}\n\n### Response(응답):"
        ),
    }
        
        for instruction, input, output in zip(samples["instruction"], samples["input"], samples["output"]):
            user_input = input 
            response = output + tokenizer.eos_token
            conversation = PROMPT_DICT['prompt_input'].replace('{instruction}', instruction[0]).replace('{input}', user_input).replace('{response}', response) 
            batch.append(conversation)
        
        return {"content": batch}

    dataset = generate_dict(df)
    
    raw_datasets = DatasetDict()
    raw_datasets["test"] = dataset

    raw_datasets = raw_datasets.map(
        preprocess,
        batched=True,
        #remove_columns=raw_datasets["test"].column_names,
    )

    test_data = raw_datasets["test"]
    print(
        f"Size of the test set: {len(test_data)}"
    )
    print(f"A sample of test dataset: {test_data[1]}")

    return test_data


if __name__ == "__main__":

    # set base directory 
    BASE_DIR = os.path.dirname(__file__)
    
    # Confirm which GPUs are visible
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=False, default=42, help='add seed number')
    parser.add_argument('--model_path', required=False, default='', help='add pretrained model path')

    args = parser.parse_args()
   
    # set seed for reproducibility
    set_seed(args.seed)
    
    model_path = os.path.join(BASE_DIR, args.model_path)
    
    # model_path = args.model_path
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    df = pd.read_csv(os.path.join(BASE_DIR, 'data/test_data.csv'), encoding='utf-8')
    
    test_dataset = create_datasets(
        df,
        tokenizer,
    )
        
    device = "cuda" if torch.cuda.is_available else "cpu"
    model = model.to(device)
    
    # inference 
    df_submission = pd.DataFrame()
    answer_list = list()
    
    for i, test_data in enumerate(tqdm(test_dataset)): 
        text = test_data['content']
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
        # Remove 'token_type_ids' if present 
        model_inputs.pop('token_type_ids', None)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=len(test_data['input']),
                eos_token_id=tokenizer.eos_token_id, 
                pad_token_id=tokenizer.pad_token_id
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        
        # create submission.csv 
        answer = response.split(args.response_split)[1].strip()
        answer_list.append(answer)
        

    df_submission['ID'] = pd.read_csv('./data/sample_submission.csv')['ID']
    df_submission['output'] = answer_list 
    
    df_submission.to_csv(os.path.join(BASE_DIR, 'submission.csv'), index=False)
