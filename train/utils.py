import torch
from torch.nn.utils.rnn import pad_sequence
from config import DEVICE
import wandb

def text_2_ids_and_attention_mask(tokenizer, input_txt, truncate=False):
    txt = input_txt
    res = tokenizer(txt, return_tensors="pt")

    if truncate:
        return res.input_ids[:, 1:], res.attention_mask[:, 1:]

    return res.input_ids, res.attention_mask

def prompt_template_fn(prompt="Describe the sound of the given file"):
    system_message = "You are a helpful AI who follows instruction carefully"
    prompt_prefix = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {prompt}
    """
    #Llama-3 prompt template
    # https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
    return prompt_prefix

def end_template():
    #llama-3 end tokens
    return """
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

def pad_sequence_start(sequences, batch_first=False, padding_value=0.0):
    max_length = max(seq.size(0) for seq in sequences)
    padded_seqs = []
    
    for seq in sequences:
        pad_length = max_length - seq.size(0)
        padding = torch.full((pad_length,) + seq.size()[1:], padding_value, dtype=seq.dtype, device=seq.device)
        padded_seq = torch.cat([padding, seq], dim=0)
        padded_seqs.append(padded_seq)
    
    if batch_first:
        return torch.stack(padded_seqs, dim=0)
    else:
        return torch.stack(padded_seqs, dim=1)

def move_to_device(batch, device=DEVICE):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

def log_results_to_wandb(results):
    table = wandb.Table(columns=["Ground Truth", "Generated"])
    for result in results:
        table.add_data(result["Ground Truth"], result["Generated"])
    wandb.log({"Caption Comparison": table})