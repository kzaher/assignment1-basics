import regex as re
from collections import abc
import pickle
from cs336_basics import bpe_constants
import heapq


class BpeTokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab_ = vocab.copy()
        self.merges_ = merges.copy()
        self.special_tokens_ = special_tokens or []
        self.special_tokens_bytes_ = [special_token.encode('utf-8') for special_token in self.special_tokens_]
        self.inverse_vocab_ = {v: k for k, v in self.vocab_.items()}
        for special_token_bytes in self.special_tokens_bytes_:
          if special_token_bytes in self.inverse_vocab_:
              continue
          self.inverse_vocab_[special_token_bytes] = len(self.vocab_)
          self.vocab_[len(self.vocab_)] = special_token_bytes
        self.token_merges_ = {
            (
                self.inverse_vocab_[merge[0]],
                self.inverse_vocab_[merge[1]],
            ): (index, self.inverse_vocab_[merge[0] + merge[1]])
            for index, merge in enumerate(self.merges_)
        }

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.loads(f.read())
        with open(merges_filepath, "wb") as f:
            merges = pickle.loads(f.read())
        return BpeTokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def persist(self, vocab_filepath, merges_filepath):
        with open(vocab_filepath, "wb") as f:
            f.write(pickle.dumps(self.vocab_))

        values = list(self.vocab_.values())
        values.sort(key=lambda x: len(x))
        with open(vocab_filepath + ".txt", "wt") as f:
            for v in values:
                f.write(repr(v))
                f.write("\n")

        with open(merges_filepath, "wb") as f:
            f.write(pickle.dumps(self.merges_))

    def bytes_for_tokens(self, tokens):
        return [self.vocab_[token] for token in tokens]

    class WorkQueue:
        def __init__(self, token_merges: dict[tuple[int, int], tuple[int, int]]):
            self.token_merges_ = token_merges
            self.queue_ = list[tuple[int, int, int, int, int, int]]()

        def maybe_push(self, i: int, j: int, tokens: list[int]):
            potential_merge = self.token_merges_.get((tokens[i], tokens[j]))
            if potential_merge is None:
                return
            heapq.heappush(
                self.queue_,
                (potential_merge[0], potential_merge[1], i, j, tokens[i], tokens[j]),
            )

        def pop(self) -> tuple[(int, int, int, int, int, int)] | None:
            if not self.queue_:
                return None
            return heapq.heappop(self.queue_)

    def encode_pretoken_(self, pretoken: bytes, queue: WorkQueue) -> list[int]:
        tokens = [self.inverse_vocab_[pretoken[i : i + 1]] for i in range(len(pretoken))]
        for i in range(len(tokens) - 1):
            queue.maybe_push(i, i + 1, tokens)

        while (next := queue.pop()) != None:
            _, replacement, i, i_next, token1, token2 = next
            if not (tokens[i] == token1 and tokens[i_next] == token2):
                continue
            tokens[i] = replacement
            tokens[i_next] = -1
            j = i - 1
            while j >= 0:
                if tokens[j] == -1:
                    j -= 1
                    continue
                queue.maybe_push(j, i, tokens)
                break
            j = i + 1
            while j < len(tokens):
                if tokens[j] == -1:
                    j += 1
                    continue
                queue.maybe_push(i, j, tokens)
                break

        return_value = [i for i in tokens if i >= 0]
        decoded = b"".join([self.vocab_[i] for i in return_value])
        assert decoded == pretoken, f"{decoded} != {pretoken}"
        return return_value

    def pretokenize_and_encode_(self, text: str, queue: WorkQueue) -> list[int]:
        pretokens = [
            pretoken.encode("utf-8") for pretoken in re.findall(bpe_constants.PAT, text)
        ]
        tokens = [
            token
            for pretoken in pretokens
            for token in self.encode_pretoken_(pretoken, queue)
        ]
        return tokens

    def encode(self, text: str) -> list[int]:
        tokens: list[int] = []
        queue = BpeTokenizer.WorkQueue(self.token_merges_)
        processed_until_index = 0

        def text_matches(at_index: int, value: str):
          for j in range(len(value)):
            if value[j] != text[at_index + j]:
              return False
          return True

        # Because of overlapping special tokens.
        special_tokens_by_length = list(enumerate(self.special_tokens_))
        special_tokens_by_length.sort(key=lambda x: len(x[1]), reverse=True)

        for i in range(len(text)):
          if i < processed_until_index:
            continue
          for special_token_index, special_token in special_tokens_by_length:
            if len(text) - i < len(special_token):
              continue
            if text_matches(at_index=i, value=special_token):
              tokens.extend(self.pretokenize_and_encode_(text[processed_until_index:i], queue))
              processed_until_index = i + len(special_token)
              tokens.append(self.inverse_vocab_[self.special_tokens_bytes_[special_token_index]])
              break
        tokens.extend(self.pretokenize_and_encode_(text[processed_until_index:len(text)], queue))
        return tokens
        

    def encode_iterable(self, iterable: abc.Iterable[str]) -> abc.Iterator[int]:
        for i in iterable:
            for token in self.encode(i):
                yield token

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab_[id] for id in ids).decode("utf-8", errors="replace")
