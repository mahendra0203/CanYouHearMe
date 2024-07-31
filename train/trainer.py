from datasets import Dataset
from transformers import Trainer, TrainingArguments
from typing import Dict, List, Union
from tqdm import tqdm
import torch
from utils import log_results_to_wandb, move_to_device
from config import DEVICE

def run_inference(model, dataloader, tokenizer, device, num_samples=10):
    model.eval()
    results = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Running inference")):
            if i >= num_samples:
                break
            
            batch = move_to_device(batch, device)
            
            outputs, audio_len = model(**batch)
            prompt_ids = batch['prompt_ids']
            end_prompt_ids = batch['end_prompt_ids']
            caption_ids = batch['caption_ids']
            
            prompt_ids_seq = prompt_ids.shape[1]
            end_prompt_ids_seq = end_prompt_ids.shape[1]
            audio_seq = audio_len
            logits_start = prompt_ids_seq + audio_seq + end_prompt_ids_seq
            
            logits = outputs.logits
            op_logits = logits[:, logits_start:-1, :].contiguous()
            caption_labels = caption_ids[:, 1:].contiguous()
            
            sampled = torch.multinomial(op_logits[:, -1, :].softmax(dim=-1), caption_labels.shape[1])
            
            ground_truth = tokenizer.batch_decode(batch['caption_ids'], skip_special_tokens=True)
            generated = tokenizer.batch_decode(sampled,skip_special_tokens=True)
            
            for gt, gen in zip(ground_truth, generated):
                results.append({"Ground Truth": gt, "Generated": gen})
    
    return results

class MultiModalDataset(Dataset):
    def __init__(self, dataloader):
        self.dataloader = dataloader
    
    def __len__(self):
        return len(self.dataloader.dataset)

    def __getitem__(self, idx):
        return self.dataloader.dataset[idx]

class MultiModalTrainer(Trainer):
    def __init__(self, test_dataloader, tokenizer,device,test_steps=500,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_dataloader = test_dataloader
        self.test_step = 0
        self.tokenizer = tokenizer
        self.test_steps = test_steps
        self.device = device
        self.model = kwargs['model']
        
    def move_to_device(self, batch, device):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
    def training_step(self, model, inputs):
        """
        Perform a training step and run test loop every 500 steps.
        """
        # Perform regular training step
        loss = super().training_step(model, inputs)

        # Increment test step counter
        self.test_step += 1

        # Run test loop every 500 steps
        # for test purposes logging at every step
        if self.test_step > 0 and self.test_step % self.test_steps == 0:
            self.run_test_loop(DEVICE)
        
        return loss
        
    def run_test_loop(self, device):
        inference_results = run_inference(self.model, 
                                          self.test_dataloader, 
                                          self.tokenizer, 
                                          device, 
                                          num_samples=2)
        log_results_to_wandb(inference_results)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs, audio_seq = model(**inputs)
        loss = self.calculate_loss(outputs, inputs, audio_seq)
        return (loss, outputs) if return_outputs else loss

    def calculate_loss(self, outputs, batch, audio_seq):
        logits = outputs.logits
        prompt_ids = batch['prompt_ids']
        end_prompt_ids = batch['end_prompt_ids']
        caption_ids = batch['caption_ids']
        
        prompt_ids_seq = prompt_ids.shape[1]
        end_prompt_ids_seq = end_prompt_ids.shape[1]
        logits_start = prompt_ids_seq + audio_seq + end_prompt_ids_seq

        op_logits = logits[:, logits_start:-1, :].contiguous()
        caption_labels = caption_ids[:, 1:].contiguous()
        
        if op_logits.shape[1] != caption_labels.shape[1]:
            raise ValueError(f"Shape mismatch: op_logits {op_logits.shape}, caption_labels {caption_labels.shape}")

        loss = torch.nn.functional.cross_entropy(
            op_logits.view(-1, op_logits.shape[-1]), caption_labels.view(-1)
        )
        return loss

    def evaluate(
            self,
            eval_dataset = None,
            ignore_keys = None,
            metric_key_prefix: str = "eval",
        ):
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            # Perform decoding and loss calculations here
            # print(self.model)
            metrics = {'wer': 1.0}
            # Validation
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in tqdm(eval_dataloader, desc="Validating"):
                    batch = self.move_to_device(batch, self.device)
                    # batch = {k: v.to(device) for k, v in batch.items()}
                    
                    outputs, audio_seq = self.model(**batch)
                    # loss = outputs.loss
                    # outputs = model(**batch)
                    loss = self.calculate_loss(outputs, batch, audio_seq)
                    # loss = calculate_loss(outputs, batch, audio_seq)
                    total_val_loss += loss.item()
                
            avg_val_loss = total_val_loss / len(eval_dataloader)
            print(f"Average Validation Loss: {avg_val_loss:.4f}")
            print()
        
            # Log metrics to wandb
            metrics = {
                "eval_loss": avg_val_loss
            }
            print(metrics)
            return metrics
