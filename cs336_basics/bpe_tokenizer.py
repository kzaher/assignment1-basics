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
        self._vocab = vocab.copy()
        self._merges = merges.copy()
        self._special_tokens = special_tokens or []
        self._special_tokens_bytes = [special_token.encode('utf-8') for special_token in self._special_tokens]
        self._inverse_vocab = {v: k for k, v in self._vocab.items()}
        for special_token_bytes in self._special_tokens_bytes:
          if special_token_bytes in self._inverse_vocab:
              continue
          self._inverse_vocab[special_token_bytes] = len(self._vocab)
          self._vocab[len(self._vocab)] = special_token_bytes
        self._token_merges = {
            (
                self._inverse_vocab[merge[0]],
                self._inverse_vocab[merge[1]],
            ): (index, self._inverse_vocab[merge[0] + merge[1]])
            for index, merge in enumerate(self._merges)
        }
        self._has_special_tokens = set[str]()
        for special_token in self._special_tokens:
            self._has_special_tokens.add(special_token[0])


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.loads(f.read())
        with open(merges_filepath, "rb") as f:
            merges = pickle.loads(f.read())
        return BpeTokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def persist(self, vocab_filepath, merges_filepath):
        with open(vocab_filepath, "wb") as f:
            f.write(pickle.dumps(self._vocab))

        values = list(self._vocab.values())
        values.sort(key=lambda x: len(x))
        with open(vocab_filepath + ".txt", "wt") as f:
            for v in values:
                f.write(repr(v))
                f.write("\n")

        with open(merges_filepath, "wb") as f:
            f.write(pickle.dumps(self._merges))

    def bytes_for_tokens(self, tokens):
        return [self._vocab[token] for token in tokens]

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
        tokens = [self._inverse_vocab[pretoken[i : i + 1]] for i in range(len(pretoken))]
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
        decoded = b"".join([self._vocab[i] for i in return_value])
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
        queue = BpeTokenizer.WorkQueue(self._token_merges)
        processed_until_index = 0

        def text_matches(at_index: int, value: str):
          for j in range(len(value)):
            if value[j] != text[at_index + j]:
              return False
          return True

        # Because of overlapping special tokens.
        special_tokens_by_length = list(enumerate(self._special_tokens))
        special_tokens_by_length.sort(key=lambda x: len(x[1]), reverse=True)

        for i in range(len(text)):
          if i < processed_until_index:
            continue
          if text[i] not in self._has_special_tokens:
              continue
          for special_token_index, special_token in special_tokens_by_length:
            if len(text) - i < len(special_token):
              continue
            if text_matches(at_index=i, value=special_token):
              tokens.extend(self.pretokenize_and_encode_(text[processed_until_index:i], queue))
              processed_until_index = i + len(special_token)
              tokens.append(self._inverse_vocab[self._special_tokens_bytes[special_token_index]])
              break
        tokens.extend(self.pretokenize_and_encode_(text[processed_until_index:len(text)], queue))
        return tokens
        

    def encode_iterable(self, iterable: abc.Iterable[str]) -> abc.Iterator[int]:
        for i in iterable:
            for token in self.encode(i):
                yield token

    def decode(self, ids: list[int]) -> str:
        return b"".join(self._vocab[id] for id in ids).decode("utf-8", errors="replace")
