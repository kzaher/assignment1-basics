from cs336_basics.train_bpe import train_bpe

train_bpe(
    # input_path="/workspace/data/TinyStoriesV2-GPT4-train.txt",
    input_path="/workspace/data/TinyStoriesV2-GPT4-train.txt",
    vocab_size=100_000,
    special_tokens=["<|endoftext|>"],
    num_processes=4,
)
