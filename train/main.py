import os
import torch
from huggingface_hub import login
import wandb

from transformers import TrainingArguments
from datasets import load_from_disk

import config
from data_processing import load_and_prepare_dataset, prepare_dataset, create_dataloaders
from model import MultiModalLLM
from utils import move_to_device, move_model_to_gpu
from trainer import MultiModalTrainer

from data_processing import tokenizer, collate_fn

def main():
    print(config.LEARNING_RATE)
    print(config.WANDB_PROJECT)
    print(config.WANDB_RUN_NAME)

    #TODO: add a better way to do login, maybe environment variables 
    login(new_session=False)
    wandb.login()

    wandb.init(project=config.WANDB_PROJECT, name=config.WANDB_RUN_NAME)

    # Load and prepare datasets
    train_dataset, val_dataset, test_dataset = load_and_prepare_dataset()
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Prepare datasets
    train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names, num_proc=config.NUM_WORKERS)
    val_dataset = val_dataset.map(prepare_dataset, remove_columns=val_dataset.column_names, num_proc=config.NUM_WORKERS)
    test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names, num_proc=config.NUM_WORKERS)

    train_dataset.save_to_disk('/home/mapped_dataset/train/')
    val_dataset.save_to_disk('/home/mapped_dataset/val/')
    test_dataset.save_to_disk('/home/mapped_dataset/test/')

    ####Only for the second or subsequent tries, dont have do the mapping again, it takes a while
    # train_dataset = load_from_disk('/home/mapped_dataset/train/')
    # val_dataset = load_from_disk('/home/mapped_dataset/val/')
    # test_dataset = load_from_disk('/home/mapped_dataset/test/')
    ###Second try end
    # Create dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(train_dataset, val_dataset, test_dataset)

    #Create the model
    model = MultiModalLLM(config.WHISPER_MODEL_NAME, config.LLM_MODEL_NAME)
    model = move_model_to_gpu(model)

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
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        test_dataloader=test_dataloader,
        tokenizer=tokenizer,
        test_steps=config.TEST_STEPS,
        device=config.DEVICE,
    )

# # Initialize Trainer
# trainer = MultiModalTrainer(
#         model=model,
#         args=training_args,
#         data_collator=collate_fn,
#         train_dataset=new_train_dataset,
#         eval_dataset=new_val_dataset,
#         test_dataloader=test_dataloader,
#         tokenizer=tokenizer,
#         test_steps=500
#     )
    # Train the model
    # trainer.train(resume_from_checkpoint=True)
    trainer.train()

    # Save the final model
    trainer.save_model()
    trainer.save_state()

    # Finish the wandb run
    wandb.finish()



if __name__ == "__main__":
    main()