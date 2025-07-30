import regex as re
from cs336_basics import pretokenization
from collections import Counter
from collections import defaultdict
from collections import abc
import multiprocessing
import os
import heapq
from functools import total_ordering


def get_pretokens(input_path: str, start: int, end: int) -> Counter[bytes]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(input_path, "rb") as f:
        f.seek(start)
        documents = (
            f.read(end - start).decode("utf-8", errors="ignore").split("<|endoftext|>")
        )
    pretokens = Counter[bytes]()
    for document in documents:
        pretokens.update(
            [pretoken.encode("utf-8") for pretoken in re.findall(PAT, document)]
        )
    return pretokens


def get_pretokens_kw_args(kw_args) -> Counter[bytes]:
    return get_pretokens(**kw_args)


@total_ordering
class MaxQueue:
    def __init__(
        self,
        total_count: int,
        raw_string_tuple: tuple[bytes, bytes],
        reference_pair: tuple[int, int],
    ):
        self.total_count_ = total_count
        self.raw_string_tuple_ = raw_string_tuple
        self.reference_pair_ = reference_pair

    def __lt__(self, other):
        if self.total_count_ > other.total_count_:
            return True
        if self.total_count_ < other.total_count_:
            return False
        return self.raw_string_tuple_ > other.raw_string_tuple_

    def __eq__(self, other):
        return (
            self.total_count_ == other.total_count_
            and self.raw_string_tuple_ == other.raw_string_tuple_
        )

    def __repr__(self):
        return repr((self.total_count_, self.raw_string_tuple_, self.reference_pair_))


def merge_tokens(
    pretoken_counts: dict[tuple[int, ...], int],
    vocabulary: dict[int, bytes],
    remaining_vocab_size: int,
) -> list[tuple[bytes, bytes]]:
    # Larger first.
    queue: list[MaxQueue] = []

    def add_to_queue(count: int, token_pair: tuple[int, int]):
        heapq.heappush(
            queue,
            MaxQueue(
                total_count=count,
                raw_string_tuple=(vocabulary[token_pair[0]], vocabulary[token_pair[1]]),
                reference_pair=token_pair,
            ),
        )

    def pop_biggest_count() -> MaxQueue:
        return heapq.heappop(queue)

    distinct_merged_pretokens = list(pretoken_counts.keys())
    pretoken_counts_by_reference = {
        index: pretoken_counts[distinct_pretokens]
        for index, distinct_pretokens in enumerate(distinct_merged_pretokens)
    }

    # Pair to list index.
    optimistic_references: dict[tuple[int, int], set[int]] = defaultdict(set)

    def get_pairs_of_tokens(
        tokenized_pretoken: tuple[int, ...],
    ) -> abc.Iterator[tuple[int, int]]:
        for i in range(len(tokenized_pretoken) - 1):
            yield (tokenized_pretoken[i], tokenized_pretoken[i + 1])

    token_pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    for pretoken_index, tokenized_pretoken in enumerate(distinct_merged_pretokens):
        for token_pair in get_pairs_of_tokens(tokenized_pretoken):
            optimistic_references[token_pair].add(pretoken_index)
            token_pair_counts[token_pair] += pretoken_counts_by_reference[
                pretoken_index
            ]

    for merged_pair in optimistic_references:
        add_to_queue(token_pair_counts[merged_pair], merged_pair)

    def merge_pair_in_tokens(
        pretokens_unique_reference: int,
        merge_pair: tuple[int, int],
        result_token: int,
        changed_set: set[tuple[int, int]],
    ) -> tuple[int, ...]:
        merged = list(distinct_merged_pretokens[pretokens_unique_reference])
        pretoken_increment = pretoken_counts_by_reference[pretokens_unique_reference]
        for i in range(len(merged) - 1):
            # This is important because merged is reducing in size
            if i >= len(merged) - 1:
                break
            if not (merged[i] == merge_pair[0] and merged[i + 1] == merge_pair[1]):
                continue

            for j in range(max(i - 1, 0), min(i + 2, len(merged) - 1)):
                i_pair = (merged[j], merged[j + 1])
                token_pair_counts[i_pair] -= pretoken_increment
                changed_set.add(i_pair)
            merged[i : i + 2] = [result_token]
            for j in range(max(i - 1, 0), min(i + 1, len(merged) - 1)):
                i_pair = (merged[j], merged[j + 1])
                token_pair_counts[i_pair] += pretoken_increment
                changed_set.add(i_pair)
                optimistic_references[i_pair].add(pretokens_unique_reference)

        for i in range(len(merged) - 1):
            assert not (
                merged[i] == merge_pair[0] and merged[i + 1] == merge_pair[1]
            ), f"Merged={merged}, pair={merge_pair}"

        return tuple(merged)

    def merge_pair(
        pair: tuple[int, int], merged_token: int, changed_set: set[tuple[int, int]]
    ) -> int:
        total_merged = 0
        for pretokens_unique_reference in optimistic_references[pair]:
            new_value = merge_pair_in_tokens(
                pretokens_unique_reference, pair, merged_token, changed_set
            )
            total_merged += pretoken_counts_by_reference[pretokens_unique_reference] * (
                len(distinct_merged_pretokens[pretokens_unique_reference])
                - len(new_value)
            )
            distinct_merged_pretokens[pretokens_unique_reference] = new_value
        return total_merged

    merged_tokens = []
    merged_tokens_set = set()
    while remaining_vocab_size > 0 and queue:
        queue_element = pop_biggest_count()
        merged_pair = queue_element.reference_pair_
        actual_count = token_pair_counts[merged_pair]
        if actual_count != queue_element.total_count_:
            assert actual_count < queue_element.total_count_
            continue
        if queue_element.reference_pair_ in merged_tokens_set:
            continue
        merged_tokens_set.add(queue_element.reference_pair_)
        new_token = len(vocabulary)
        vocabulary[new_token] = vocabulary[merged_pair[0]] + vocabulary[merged_pair[1]]
        merged_tokens.append((vocabulary[merged_pair[0]], vocabulary[merged_pair[1]]))
        changed_set = set[tuple[int, int]]()
        total_merged = merge_pair(queue_element.reference_pair_, new_token, changed_set)
        for pair in changed_set:
            if token_pair_counts[pair] > 0:
                add_to_queue(token_pair_counts[pair], pair)
        assert (
            token_pair_counts[merged_pair] == 0
        ), f"All pairs are not merged: {token_pair_counts[merged_pair]}, {queue_element.raw_string_tuple_}"
        remaining_vocab_size -= 1

    return merged_tokens


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes=4,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_tasks = num_processes * 8
    with open(input_path, "rb") as f:
        boundaries = pretokenization.find_chunk_boundaries(
            f, num_tasks, "<|endoftext|>".encode("utf-8")
        )

    with multiprocessing.Pool(processes=num_processes) as pool:
        pretoken_counters_list = pool.map(
            get_pretokens_kw_args,
            [
                dict(input_path=input_path, start=start, end=end)
                for start, end in zip(boundaries[:-1], boundaries[1:])
            ],
        )

    pretoken_counts = Counter[bytes]()
    for counter_i in pretoken_counters_list:
        pretoken_counts = pretoken_counts + counter_i

    vocabulary = {index: bytes([index]) for index in range(256)}

    remaining_vocabulary_size = vocab_size - len(special_tokens) - len(vocabulary)
    assert remaining_vocabulary_size >= 0, "The vocabulary size is too small."

    inverse_vocabulary = {v: k for k, v in vocabulary.items()}

    all_special_token_bytes = [
        (vocab_size - len(special_tokens) + index, token.encode("utf-8"))
        for index, token in enumerate(special_tokens)
    ]
    all_special_token_bytes.sort(key=lambda x: x[1])

    def tokenize(utf8_bytes: bytes) -> tuple[int, ...]:
        return tuple([inverse_vocabulary[utf8_bytes[i : i + 1]] for i in range(len(utf8_bytes))])

    merged_tokens = merge_tokens(
        pretoken_counts={tokenize(k): v for k, v in pretoken_counts.items()},
        vocabulary=vocabulary,
        remaining_vocab_size=remaining_vocabulary_size,
    )

    for index, vocabulary_special_token in all_special_token_bytes:
        vocabulary[index] = vocabulary_special_token

    return (vocabulary, merged_tokens)
