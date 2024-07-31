import os
import torch
from huggingface_hub import login
import wandb

from transformers import TrainingArguments

import config
from data_processing import load_and_prepare_dataset, prepare_dataset, create_dataloaders
from model import MultiModalLLM
from utils import move_to_device
from trainer import MultiModalTrainer

from data_processing import tokenizer

def main():
    print(config.LEARNING_RATE)
    print(config.WANDB_PROJECT)
    print(config.WANDB_RUN_NAME)

    #TODO: add a better way to do login, maybe environment variables 
    login()
    wandb.login()
    
    # Load and prepare datasets
    train_dataset, val_dataset, test_dataset = load_and_prepare_dataset()
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Prepare datasets
    train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names, num_proc=config.NUM_WORKERS)
    val_dataset = val_dataset.map(prepare_dataset, remove_columns=val_dataset.column_names, num_proc=config.NUM_WORKERS)
    test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names, num_proc=config.NUM_WORKERS)

    # Create dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(train_dataset, val_dataset, test_dataset)

    #Create the model
    model = MultiModalLLM(config.WHISPER_MODEL_NAME, config.LLM_MODEL_NAME)
    model = move_to_device(model)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=config.WEIGHT_DECAY,
        logging_dir=config.LOGGING_DIR,
        logging_steps=config.LOGGING_STEPS,
        evaluation_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_steps=config.SAVE_STEPS,
        max_steps=config.MAX_STEPS,
        load_best_model_at_end=True,
        learning_rate=config.LEARNING_RATE,
        report_to="wandb",
    )


    # Initialize Trainer
    trainer = MultiModalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        test_dataloader=test_dataloader,
        tokenizer=tokenizer,
        test_steps=config.EVAL_STEPS,
        device=config.DEVICE,
    )

    # Train the model
    trainer.train(resume_from_checkpoint=True)

    # Save the final model
    trainer.save_model()
    trainer.save_state()

    # Finish the wandb run
    wandb.finish()



if __name__ == "__main__":
    main()