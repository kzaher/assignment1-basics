from cs336_basics.train_bpe import train_bpe

if __name__ == '__main__':
  vocabulary, merges = train_bpe(
      input_path="/workspace/data/TinyStoriesV2-GPT4-train.txt",
      vocab_size=10_000,
      special_tokens=["<|endoftext|>"],
      num_processes=4,
  )

  values = list(vocabulary.values())
  values.sort(key=lambda x: len(x))
  for i in values[-10:]:
    print(i)
  with open('./train/tiny_vocabulary.txt', 'wt') as f:
    for v in values:
      f.write(repr(v))
      f.write('\n')