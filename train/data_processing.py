import torch

from datasets import load_dataset, Audio
from torch.utils.data import DataLoader
from transformers import WhisperFeatureExtractor, WhisperProcessor, AutoTokenizer
from config import WHISPER_MODEL_NAME, LLM_MODEL_NAME, BATCH_SIZE, NUM_WORKERS, DATASET_NAME
from utils import pad_sequence_start, prompt_template_fn, end_template

feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL_NAME)
whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def load_and_prepare_dataset():
    ds = load_dataset(DATASET_NAME)
    main_dataset = ds['train'].train_test_split(test_size=0.1)
    train_val_split_dataset = main_dataset['train'].train_test_split(test_size=0.1)

    train_dataset = train_val_split_dataset['train']
    val_dataset = train_val_split_dataset['test']
    test_dataset = main_dataset['test']

    for dataset in [train_dataset, val_dataset, test_dataset]:
        dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))

    return train_dataset, val_dataset, test_dataset

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    
    token_result = tokenizer(batch["caption"], padding=True, truncation=True)
    batch["caption_ids"] = token_result.input_ids
    batch["caption_attention_mask"] = token_result.attention_mask

    prompt_tokens = tokenizer(prompt_template_fn(), padding=True, truncation=True)
    batch["prompt_ids"] = prompt_tokens.input_ids
    batch["prompt_attention_mask"] = prompt_tokens.attention_mask
    
    end_prompt_tokens = tokenizer(end_template(), padding=True, truncation=True)
    batch["end_prompt_ids"] = end_prompt_tokens.input_ids
    batch["end_prompt_attention_mask"] = end_prompt_tokens.attention_mask

    return batch

def collate_fn(batch):
    input_features = [whisper_processor.feature_extractor.pad({"input_features": item['input_features']}, return_tensors="pt").input_features for item in batch]

    caption_ids = [item['caption_ids'] for item in batch]
    caption_attention_mask = [item['caption_attention_mask'] for item in batch]
    prompt_ids = [item['prompt_ids'] for item in batch]
    prompt_attention_mask = [item['prompt_attention_mask'] for item in batch]
    end_prompt_ids = [item['end_prompt_ids'] for item in batch]
    end_prompt_attention_mask = [item['end_prompt_attention_mask'] for item in batch]

    caption_ids = pad_sequence_start([torch.tensor(x) for x in caption_ids], batch_first=True, padding_value=tokenizer.pad_token_id)
    caption_attention_mask = pad_sequence_start([torch.tensor(x) for x in caption_attention_mask], batch_first=True, padding_value=tokenizer.pad_token_id)
    prompt_ids = pad_sequence_start([torch.tensor(x) for x in prompt_ids], batch_first=True, padding_value=tokenizer.pad_token_id)
    prompt_attention_mask = pad_sequence_start([torch.tensor(x) for x in prompt_attention_mask], batch_first=True, padding_value=tokenizer.pad_token_id)
    end_prompt_ids = pad_sequence_start([torch.tensor(x) for x in end_prompt_ids], batch_first=True, padding_value=tokenizer.pad_token_id)
    end_prompt_attention_mask = pad_sequence_start([torch.tensor(x) for x in end_prompt_attention_mask], batch_first=True, padding_value=tokenizer.pad_token_id)

    input_features = torch.stack([torch.tensor(x) for x in input_features])

    return {
        'input_features': input_features,
        'caption_ids': caption_ids,
        'caption_attention_mask': caption_attention_mask,
        'prompt_ids': prompt_ids,
        'prompt_attention_mask': prompt_attention_mask,
        'end_prompt_ids': end_prompt_ids,
        'end_prompt_attention_mask': end_prompt_attention_mask
    }

def create_dataloaders(train_dataset, val_dataset, test_dataset):
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    
    return train_dataloader, val_dataloader, test_dataloader