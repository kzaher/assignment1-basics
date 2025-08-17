from cs336_basics.train_bpe import train_bpe
from cs336_basics.bpe_tokenizer import BpeTokenizer

if __name__ == "__main__":
    vocabulary, merges = train_bpe(
        input_path="./data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )

    tokenizer = BpeTokenizer(vocab=vocabulary, merges=merges)
    tokenizer.persist(
        vocab_filepath="./train/tiny_vocabulary", merges_filepath="./train/tiny_merges"
    )
