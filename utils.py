import os
import torch
import transformers
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from sklearn.metrics import precision_score, recall_score, f1_score

from peft import LoraConfig


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
    
    def generate_dict(df) : 
    
        instruction_list = [ [open('./data/instruction.txt').read()] for _ in range(len(df)) ] 
        input_list = df['input']
        output_list = df['output']
    
        dataset_dict = {'instruction' : instruction_list , 'input' : input_list , 'output' : output_list} 
        dataset = Dataset.from_dict(dataset_dict)
    
        return dataset 

    dataset = generate_dict(df) 
    raw_datasets = DatasetDict()
    datasets = dataset.train_test_split(test_size = 0.2,
                                        shuffle= True , 
                                        seed = 42)
    
    raw_datasets['train'] = datasets['train']
    raw_datasets['test'] = datasets['test'] 
    
    raw_datasets = raw_datasets.map(
        preprocess,
        batched = True,
        remove_columns=raw_datasets['train'].column_names
    )
    
    train_data = raw_datasets["train"]
    valid_data = raw_datasets["test"]
    print(
        f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
    )
    print(f"A sample of train dataset: {train_data[0]}")

    return train_data, valid_data


def create_and_prepare_model(args, data_args, training_args):
    
    
    if args.use_unsloth:
        from unsloth import FastLanguageModel
    
    bnb_config = None
    quant_storage_dtype = None

    # 분산 훈련 사용시, unsloth 가 지원되지 않기 때문에 예외처리리
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and args.use_unsloth
    ):
        raise NotImplementedError("Unsloth is not supported in distributed training")

    #4비트 양자화 설정할 경우 
    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
        
        #8비트 양자화 설정정 
        elif args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)

    if args.use_unsloth:
        # Load model
        model, _ = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            dtype=None,
            load_in_4bit=args.use_4bit_quantization,
        )
    else:
        torch_dtype = (
            quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
            torch_dtype=torch_dtype,
        )

    peft_config = None #PEFT 초기화
    
    if args.use_peft_lora and not args.use_unsloth:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(",") 
            if args.lora_target_modules != "all-linear" else args.lora_target_modules,
        )

 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = 'right' #패딩을 오른쪽에 추가

    #unsloth를 사용할 경우 
    if args.use_unsloth:
        # Do model patching and add fast LoRA weights
        model = FastLanguageModel.get_peft_model(
            model,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            random_state=training_args.seed,
        )

    return model, peft_config, tokenizer


def compute_metrics(eval_pred , tokenizer) : 
    
    predictions , labels = eval_pred 
    
    decoded_preds = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
    decoded_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

    # Concatenate all predictions and labels into single strings
    all_preds = "".join(decoded_preds)
    all_labels = "".join(decoded_labels)
     
        # Ensure predictions and labels are the same length
    if len(all_preds) != len(all_labels):
        min_len = min(len(all_preds), len(all_labels))
        all_preds = all_preds[:min_len]
        all_labels = all_labels[:min_len]

    # Calculate Precision, Recall, and F1 Score at character level
    precision = precision_score(list(all_labels), list(all_preds), average='micro', zero_division=0)
    recall = recall_score(list(all_labels), list(all_preds), average='micro', zero_division=0)
    f1 = f1_score(list(all_labels), list(all_preds), average='micro', zero_division=0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    