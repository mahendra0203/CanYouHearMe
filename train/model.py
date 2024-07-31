import torch
import torch.nn as nn
from transformers import WhisperModel, AutoModelForCausalLM, BitsAndBytesConfig
from config import WHISPER_MODEL_NAME, LLM_MODEL_NAME

class TunableWhisperProjection(nn.Module):
    def __init__(self, input_embedding_size=1280, output_embedding_size=4096):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(250)
        self.proj = nn.Linear(input_embedding_size, output_embedding_size, bias=False)
        self.ln1 = nn.LayerNorm(input_embedding_size)

    def forward(self, whisper_output):
        pooled = self.pool(whisper_output.transpose(-2, -1))
        normalized = self.ln1(pooled.transpose(-2, -1))
        projected = self.proj(normalized)
        return projected

class MultiModalLLM(nn.Module):
    def __init__(self, whisper_model_name=WHISPER_MODEL_NAME, llm_model_name=LLM_MODEL_NAME):
        super().__init__()
        self.audio_encoder = WhisperModel.from_pretrained(whisper_model_name).get_encoder()
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name, quantization_config=bnb_config)
        for param in self.llm.parameters():
            param.requires_grad = False

        #input_embedding_size is whisper-large-v2 op and output_embedding_size is the context_window to llama-3
        self.projection = TunableWhisperProjection(input_embedding_size=1280, output_embedding_size=4096)

    def forward(self, input_features, prompt_ids, prompt_attention_mask, 
                end_prompt_ids, end_prompt_attention_mask, 
                caption_ids, caption_attention_mask):
        
        audio_outputs = self.audio_encoder.forward(input_features).last_hidden_state
        projected_audio = self.projection(audio_outputs)

        if self.llm.model.name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct":
            cap_embeds = self.llm.model.embed_tokens(caption_ids)
            prompt_embeds = self.llm.model.embed_tokens(prompt_ids)
            end_prompt_embeds = self.llm.model.embed_tokens(end_prompt_ids)
        else:
            cap_embeds = self.llm.model.get_decoder().embed_tokens(caption_ids)
            prompt_embeds = self.llm.model.get_decoder().embed_tokens(prompt_ids)
            end_prompt_embeds = self.llm.model.get_decoder().embed_tokens(end_prompt_ids)
        
        bs, audio_seq = projected_audio.shape[:2]
        
        inputs_embeds = torch.concat(
            (
                prompt_embeds,
                projected_audio.to(cap_embeds.dtype),
                end_prompt_embeds,
                cap_embeds,
            ),
            dim=1,
        )
        
        attention_mask = torch.concat(
            (
                prompt_attention_mask,
                torch.ones(bs, audio_seq).to(caption_ids.device),
                end_prompt_attention_mask,
                caption_attention_mask,
            ),
            dim=1,
        )
        
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        return outputs, audio_seq