import time
from cs336_basics.bpe_tokenizer import BpeTokenizer

if __name__ == '__main__':
  
  tokenizer = BpeTokenizer.from_files('./train/tiny_vocabulary', './train/tiny_merges')
  with open('./data/TinyStoriesV2-GPT4-train.txt', 'rt') as f:
    data = f.read()
  start = time.perf_counter()
  tokenizer.encode(data)
  end = time.perf_counter()
  print(f'Time to encode {len(data.encode('utf-8'))} is {end - start:.3} seconds')