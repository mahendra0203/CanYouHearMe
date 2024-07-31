import subprocess, os
from pathlib import Path
import multiprocessing as mp
from functools import partial
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset, Audio

import yt_dlp

def download_youtube_audio(youtube_id,
                           start_time,
                           end_time,
                           output_file,
                           num_attempts=5,
                           url_base='https://www.youtube.com/watch?v='):
  status = False
  command = f"""
        yt-dlp --quiet --no-warnings -x --audio-format wav --postprocessor-args "-ar 16000" -o "{output_file}" --download-sections "*{start_time}-{end_time}" {url_base}{youtube_id}""".strip()
  attempts = 0
  while True:
    try:
      output = subprocess.check_output(command, shell=True,
                                                stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
      attempts += 1
      if attempts == num_attempts:
        return status, err.output
      else:
        break

    # Check if the video was successfully saved.
    status = os.path.exists(output_file)
    return status, 'Downloaded'
  


def process_item(output_dir, item):
  youtube_id = item['ytid']
  start_time = item['start_s']
  end_time = item['end_s']
  caption = item['caption']
  asp_list = item['aspect_list']
  output_file = Path(output_dir) / f"yt_id_{youtube_id}_{start_time}_{end_time}.wav"
  download_result = download_youtube_audio(youtube_id, start_time, end_time, str(output_file))
  if download_result:
    status, message = download_result
    if status:
      return {
        'audio': str(output_file),
        'caption': caption,
        'youtube_id': youtube_id,
        'start_time': start_time,
        'end_time': end_time,
        'aspect_list': asp_list
        }
    else:
      print(f"Failed to download {youtube_id}: {message}")
  else:
    print(f"Download failed for  {youtube_id}")
  
  return None

def create_audio_caption_dataset(data_split, output_dir, sample_range=range(0,10)):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    new_dataset = []

    samples = data_split.select(sample_range)
    # Create a partial function with fixed output_dir
    process_item_partial = partial(process_item, output_dir)

    # Use multiprocessing to process items in parallel
    print(f"Processing {len(samples)} samples")
    print(f"Using {mp.cpu_count()} processes")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_item_partial, samples),
                            total=len(samples), 
                            desc="Processing samples"))
        
    # Filter out None results (failed downloads)
    new_dataset = [item for item in results if item is not None]
    # for item in tqdm(samples, desc="Processing samples"):
        
    #     download_data = process_item(item, output_dir)
    #     if download_data:
    #         new_dataset.append(download_data)

    dataset = Dataset.from_list(new_dataset)
    dataset = dataset.cast_column("audio", Audio())

    return dataset


if __name__ == '__main__':
  dataset = load_dataset("google/MusicCaps")
  len(dataset['train'])

  dataset = dataset['train']

  #Local path on my system
  output_dir = os.path.join('/Users/m/code/multimodal-llm', "new_full_dataset/musiccaps_audio_data")

  print(f"train_len: {len(dataset)}")

  full_dataset_range = range(0, len(dataset))

  print("Creating the full dataset")
  full_dataset = create_audio_caption_dataset(dataset, output_dir, sample_range=full_dataset_range)

  print(f"New val dataset size: {len(full_dataset)}")
   #Local path on my system
  full_train_path = os.path.join('/Users/m/code/multimodal-llm', "new_full_dataset/train/train_musiccaps_audio_data")

  print("save the full dataset to disk")
  full_dataset.save_to_disk(full_train_path)

  print(f"Full dataset saved to :{full_train_path}")

  print("save the full dataset to huggingface hub")
  full_dataset.push_to_hub("mahendra0203/musiccaps_processed_full", split="train", private=True)


  # test_size = 0.1
  # val_size = 0.1
  # seed = 1835

  # # split_dataset = dataset['train'].train_test_split(test_size=test_size, seed=1835)

  # split_main_dataset = dataset['train'].train_test_split(test_size=test_size, seed=seed)
              
  # # Now, split the remaining data into train and validation
  # split_train_val_dataset = split_main_dataset['train'].train_test_split(test_size=val_size/(1-test_size),  # Adjust validation size
  #                 seed=seed
  #             )
  # print(f"train_len: {len(split_train_val_dataset['train'])}")
  # print(f"val_len: {len(split_train_val_dataset['test'])}")
  # print(f"test_len: {len(split_main_dataset['test'])}")


  # output_dir = os.path.join('/Users/m/code/multimodal-llm', "full_dataset/musiccaps_audio_data")


  # hf_train_dataset = split_train_val_dataset['train']
  # hf_val_dataset = split_train_val_dataset['test']
  # hf_test_dataset = split_main_dataset['test']

  # training_range = range(0, len(hf_train_dataset))
  # val_range = range(0, len(hf_val_dataset))
  # test_range = range(0, len(hf_test_dataset))

  # # training_range = range(0, 10)
  # # val_range = range(0, 10)
  # # test_range = range(0, 10)

  # # print("Creating train dataset")
  # # train_dataset = create_audio_caption_dataset(hf_train_dataset, output_dir, sample_range=training_range)
  # # for i, entry in enumerate(train_dataset.select(range(min(2, len(train_dataset))))):
  # #     print(f"\nEntry {i + 1}:")
  # #     for key, value in entry.items():
  # #         print(f"  {key}: {value}")

  # # # Save the train dataset
  # # train_path = os.path.join('/Users/m/code/multimodal-llm', "full_dataset/train/train_musiccaps_audio_data")
  # # train_dataset.save_to_disk(train_path)
  # # print(f"training dataset saved to :{train_path}")


  # # Create val dataset 
  # print("Creating val dataset")
  # val_dataset = create_audio_caption_dataset(hf_val_dataset, output_dir, sample_range= val_range)
  # print(f"New val dataset size: {len(val_dataset)}")
  # val_path = os.path.join('/Users/m/code/multimodal-llm', "full_dataset/val/val_musiccaps_audio_data")
  # val_dataset.save_to_disk(val_path)
  # print(f"val dataset saved to :{val_path}")


  # #create test dataset
  # print("Creating test dataset")
  # test_dataset = create_audio_caption_dataset(hf_test_dataset, output_dir, sample_range=test_range)
  # print(f"New test dataset size: {len(test_dataset)}")
  # test_path = os.path.join('/Users/m/code/multimodal-llm', "full_dataset/test/test_musiccaps_audio_data")
  # test_dataset.save_to_disk(test_path)
  # print(f"test dataset saved to :{test_path}")