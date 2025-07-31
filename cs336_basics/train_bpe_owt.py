from cs336_basics.train_bpe import train_bpe
from cs336_basics.bpe_tokenizer import BpeTokenizer

if __name__ == "__main__":
    vocabulary, merges = train_bpe(
        input_path="./data/owt_train.txt",
        vocab_size=32_000,
        special_tokens=["<|endoftext|>"]
    )

    tokenizer = BpeTokenizer(vocab=vocabulary, merges=merges)
    tokenizer.persist(
        vocab_filepath="./train/owt_vocabulary", merges_filepath="./train/owt_merges"
    )
