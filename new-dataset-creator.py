from datasets import load_dataset, concatenate_datasets, Audio


def download_alpaca_audio_test():
  print("Downloading AudioLLMs/alpaca_audio_test")
  alpaca_audio_test = load_dataset('AudioLLMs/alpaca_audio_test')
  alpaca_audio_test = alpaca_audio_test['test']

  def prepare_alpaca_audio_test_dataset(batch):
    audio = batch['context']

    prompt = batch['speech_instruction']

    answer = batch['answer']

    batch['audio'] = audio
    batch['prompt'] = prompt
    batch['response'] = answer

    batch['dataset_name'] = 'alpaca_audio_test'
    
    return batch
  alpaca_audio_test_dataset = alpaca_audio_test.map(prepare_alpaca_audio_test_dataset,
                                                    remove_columns=alpaca_audio_test.column_names,
                                                    num_proc=4,
                                                    load_from_cache_file=False)
  # alpaca_audio_test_dataset = alpaca_audio_test_dataset.cast_column("audio", Audio())
  return alpaca_audio_test_dataset
  
def download_dream_tts_test():
  print("Downloading AudioLLMs/dream_tts_test")
  dream_tts_test = load_dataset('AudioLLMs/dream_tts_test')
  dream_tts_test = dream_tts_test['test']

  def prepare_dream_tts_test_dataset(batch):
    audio = batch['context']

    prompt = batch['instruction']

    answer = batch['answer']

    batch['audio'] = audio
    batch['prompt'] = prompt
    batch['response'] = answer

    batch['dataset_name'] = 'dream_tts_test'
    
    return batch
  
  dream_tts_test_dataset = dream_tts_test.map(prepare_dream_tts_test_dataset,
                                              remove_columns=dream_tts_test.column_names,
                                              num_proc=4,
                                              load_from_cache_file=False)
  # dream_tts_test_dataset = dream_tts_test_dataset.cast_column("audio", Audio())
  return dream_tts_test_dataset

def download_wavcaps_test():
  print("Downloading AudioLLMs/wavcaps_test")
  wavcaps_test = load_dataset('AudioLLMs/wavcaps_test')
  wavcaps_test = wavcaps_test['test']

  def prepare_wavcaps_test_dataset(batch):
    audio = batch['context']

    prompt = batch['instruction']

    answer = batch['answer']

    batch['audio'] = audio
    batch['prompt'] = prompt
    batch['response'] = answer

    batch['dataset_name'] = 'wavcaps_test'
    
    return batch
  
  wavcaps_test_dataset = wavcaps_test.map(prepare_wavcaps_test_dataset,
                                          remove_columns=wavcaps_test.column_names,
                                          num_proc=4,
                                          load_from_cache_file=False)
  # wavcaps_test_dataset = wavcaps_test_dataset.cast_column("audio", Audio())
  return wavcaps_test_dataset

def download_clotho_aqa_test():
  print("Downloading AudioLLMs/clotho_aqa_test")
  clotho_aqa_test = load_dataset('AudioLLMs/clotho_aqa_test')
  clotho_aqa_test = clotho_aqa_test['test']

  def prepare_clotho_aqa_test_dataset(batch):
    audio = batch['context']

    prompt = batch['instruction']

    answer = batch['answer']

    batch['audio'] = audio
    batch['prompt'] = prompt
    batch['response'] = answer

    batch['dataset_name'] = 'clotho_aqa_test'
    
    return batch
  clotho_aqa_test_dataset = clotho_aqa_test.map(prepare_clotho_aqa_test_dataset,
                                                remove_columns=clotho_aqa_test.column_names,
                                                num_proc=4,
                                                load_from_cache_file=False)
  # clotho_aqa_test_dataset = clotho_aqa_test_dataset.cast_column("audio", Audio())
  return clotho_aqa_test_dataset

def download_cn_college_listen_test():
  print("Downloading AudioLLMs/cn_college_listen_test")
  cn_college_listen_test = load_dataset('AudioLLMs/cn_college_listen_test')
  cn_college_listen_test = cn_college_listen_test['test']

  def prepare_cn_college_listen_test_dataset(batch):
    audio = batch['context']

    prompt = batch['instruction']

    answer = batch['answer']

    batch['audio'] = audio
    batch['prompt'] = prompt
    batch['response'] = answer

    batch['dataset_name'] = 'cn_college_listen_test'
    
    return batch
  
  cn_college_listen_test_dataset = cn_college_listen_test.map(prepare_cn_college_listen_test_dataset,
                                                              remove_columns=cn_college_listen_test.column_names,
                                                              num_proc=4,
                                                              load_from_cache_file=False)
  # cn_college_listen_test_dataset = cn_college_listen_test_dataset.cast_column("audio", Audio())
  return cn_college_listen_test_dataset

def download_audiocaps_test():
  print("Downloading AudioLLMs/audiocaps_test")
  audiocaps_test = load_dataset('AudioLLMs/audiocaps_test')
  audiocaps_test = audiocaps_test['test']

  def prepare_audiocaps_test_dataset(batch):
    audio = batch['context']

    prompt = batch['instruction']

    answer = batch['answer']

    batch['audio'] = audio
    batch['prompt'] = prompt
    batch['response'] = answer

    batch['dataset_name'] = 'audiocaps_test'
    
    return batch
  
  audiocaps_test_dataset = audiocaps_test.map(prepare_audiocaps_test_dataset,
                                              remove_columns=audiocaps_test.column_names,
                                              num_proc=4,
                                              load_from_cache_file=False)
  # audiocaps_test_dataset = audiocaps_test_dataset.cast_column("audio", Audio())
  return audiocaps_test_dataset



def download_tedlium3_test_v2():
  print("Downloading AudioLLMs/tedlium3_test_v2")
  tedlim3_test_v2 = load_dataset('AudioLLMs/tedlium3_test_v2')
  tedlim3_test_v2 = tedlim3_test_v2['test']

  def prepare_tedlim3_test_v2_dataset(batch):
    audio = batch['context']

    prompt = batch['instruction']

    answer = batch['answer']

    batch['audio'] = audio
    batch['prompt'] = prompt
    batch['response'] = answer

    batch['dataset_name'] = 'tedlium3_test_v2'
    
    return batch
  
  tedlim3_test_v2 = tedlim3_test_v2.map(prepare_tedlim3_test_v2_dataset,
                                        remove_columns=tedlim3_test_v2.column_names,
                                        num_proc=4,
                                        load_from_cache_file=False)
  # tedlim3_test_v2 = tedlim3_test_v2.cast_column("audio", Audio())
  return tedlim3_test_v2
                                        
def download_peoples_speech_test_v2():
  print("Downloading AudioLLMs/peoples_speech_test_v2")
  peoples_speech_test_v2 = load_dataset('AudioLLMs/peoples_speech_test_v2')
  peoples_speech_test_v2 = peoples_speech_test_v2['test']
  def prepare_peoples_speech_test_v2_dataset(batch):
    audio = batch['context']

    prompt = batch['instruction']

    answer = batch['answer']

    batch['audio'] = audio
    batch['prompt'] = prompt
    batch['response'] = answer

    batch['dataset_name'] = 'peoples_speech_test_v2'
    
    return batch
  
  peoples_speech_test_v2 = peoples_speech_test_v2.map(prepare_peoples_speech_test_v2_dataset,
                                                      remove_columns=peoples_speech_test_v2.column_names,
                                                      num_proc=4,
                                                      load_from_cache_file=False)
  peoples_speech_test_v2 = peoples_speech_test_v2.cast_column("audio", Audio())
  return peoples_speech_test_v2


def datasets_downloader():
  print("create datasets")
  print("Download emotions and sentiments dataset")

  print("Downloading AudioLLMs/meld_sentiment_test ")
  sentiment_test = load_dataset('AudioLLMs/meld_sentiment_test')
  def prepare_meld_sentiment_dataset(batch):
    # audio = batch['context']

    prompt = batch['instruction']

    answer = batch['answer']

    batch['audio'] = batch.pop('context')
    batch['prompt'] = prompt
    batch['response'] = answer
    
    batch['dataset_name'] = 'meld'
    
    return batch
  
  meld_dataset = sentiment_test['test'].map(prepare_meld_sentiment_dataset, 
                                          remove_columns=sentiment_test['test'].column_names, 
                                          num_proc=4,
                                          load_from_cache_file=False)
  


  print("Downloading AudioLLMs/iemocap_emotion_test")
  iemocap_emotion_test = load_dataset('AudioLLMs/iemocap_emotion_test')
  def prepare_iemocap_emotion_dataset(batch):
    audio = batch['context']

    prompt = batch['instruction']

    answer = batch['answer']

    batch['audio'] = audio
    batch['prompt'] = prompt
    batch['response'] = answer

    batch['dataset_name'] = 'iemocap_emotion'
    
    return batch
  iemocap_emotion_dataset = iemocap_emotion_test['test'].map(prepare_iemocap_emotion_dataset, 
                                                           remove_columns=iemocap_emotion_test['test'].column_names, 
                                                           num_proc=4,
                                                           load_from_cache_file=False)
  



  print("Downloading gender AudioLLMs/iemocap_gender_test")
  iemocap_gender_dataset = load_dataset("AudioLLMs/iemocap_gender_test")
  iemocap_gender_dataset = iemocap_gender_dataset['test']
  def prepare_iemocap_gender_dataset(batch):
    audio = batch['context']

    prompt = batch['instruction']

    answer = batch['answer']

    batch['audio'] = audio
    batch['prompt'] = prompt
    batch['response'] = answer

    batch['dataset_name'] = 'iemocap_gender'
    
    return batch
  iemocap_gender_dataset_formatted = iemocap_gender_dataset.map(prepare_iemocap_gender_dataset, 
                                                           remove_columns=iemocap_gender_dataset.column_names, 
                                                           num_proc=4,
                                                           load_from_cache_file=False)


  
  peoples_speech_dataset = download_peoples_speech_test_v2()

  tedlium3_test_v2 = download_tedlium3_test_v2()

  audiocaps_test = download_audiocaps_test()

  cn_college_listen_test = download_cn_college_listen_test()

  clotho_aqa_test = download_clotho_aqa_test()

  wavcaps_test = download_wavcaps_test()

  dreamtts_test = download_dream_tts_test()

  alpaca_audio_test = download_alpaca_audio_test()

  print("Concatenating all the datasets")
  full_dataset = concatenate_datasets([meld_dataset, 
                                       iemocap_emotion_dataset, 
                                       iemocap_gender_dataset_formatted, 
                                       peoples_speech_dataset,#ASR
                                       tedlium3_test_v2, #ASR
                                      audiocaps_test, #QA
                                      cn_college_listen_test, #QA
                                      clotho_aqa_test, #QA
                                      wavcaps_test, #QA
                                      dreamtts_test, #QA
                                      alpaca_audio_test, #QA with speech_instruction
                                       ])
  
  print("Casting the audio entry to Audio for the full dataset")
  full_dataset = full_dataset.cast_column("audio", Audio())

  print("uploading the full dataset to huggingface hub")
  full_dataset.push_to_hub("mahendra0203/audio_tasks", split="test", private=True)

  return "Done"

#call the function in man
def main():
  datasets_downloader()

if __name__ == '__main__':
  main()