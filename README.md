Steps to start training

**1. Install the required packages**

```
pip install --upgrade pip
pip install torch torchaudio transformers datasets datasets[audio] accelerate evaluate jiwer huggingface_hub yt-dlp tqdm librosa soundfile wandb
pip install -q -U bitsandbytes
```

**2. Add the required tokens. Huggingface and wandb**
```
export HF_TOKEN=""
export WANDB_API_KEY=""

```

**3. Make any changes required in train/config.py and then run main.py**
```
vi train/config.py
cd train

python main.py
```

