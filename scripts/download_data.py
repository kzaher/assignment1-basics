#!python3
import os
import sys
import concurrent
import concurrent.futures
from huggingface_hub import hf_hub_download

# Download the GPT-2 tokens of Fineweb10B from huggingface. This
# saves about an hour of startup time compared to regenerating them.
# def get(fname):
#     local_dir = '/workspace/data/fineweb10B'
#     if not os.path.exists(os.path.join(local_dir, fname)):
#         hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=fname,
#                         repo_type="dataset", local_dir=local_dir)
# get("fineweb_val_%06d.bin" % 0)
# num_chunks = 103 # full fineweb10B. Each chunk is 100M tokens
# with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
#     pool.map(get, ["fineweb_train_%06d.bin" % i for i in range(1, num_chunks+1)])


def get(fname):
    local_dir = "/workspace/data"
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(
            repo_id="antematerija/owt",
            filename=fname,
            repo_type="dataset",
            local_dir=local_dir,
        )


with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
    pool.map(
        get,
        [
            "owt_train.txt.tokens.vocab_size=32000.npy",
            "owt_valid.txt.tokens.vocab_size=32000.npy",
        ],
    )
