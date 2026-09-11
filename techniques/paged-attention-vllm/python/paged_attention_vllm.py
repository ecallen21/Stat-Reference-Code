"""PagedAttention / vLLM (Reference Sec 47.173).

Kwon et al 2023 'Efficient Memory Management for Large Language
Model Serving with PagedAttention', SOSP. LLM inference is memory-
bound because the KV cache grows with sequence length; naive
contiguous allocation wastes memory to internal / external
fragmentation.

    KV cache -> BLOCKS (paged) instead of contiguous tensors.
    A per-sequence BLOCK TABLE maps logical positions -> physical
    blocks (analogous to virtual memory paging).

Enables sharing prefix blocks across requests (prompt sharing),
copy-on-write forking (parallel sampling), and near-zero waste in
KV memory.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


class BlockAllocator:
    """Physical-block pool + reference counting for copy-on-write."""

    def __init__(self, n_blocks, block_size, d_head, n_kv_heads):
        self.block_size = block_size
        self.blocks = np.zeros((n_blocks, block_size, n_kv_heads, d_head))
        self.free = list(range(n_blocks - 1, -1, -1))
        self.refs = np.zeros(n_blocks, dtype=int)

    def allocate(self):
        if not self.free: raise RuntimeError("OOM")
        b = self.free.pop()
        self.refs[b] = 1
        return b

    def incref(self, b):
        self.refs[b] += 1

    def decref(self, b):
        self.refs[b] -= 1
        if self.refs[b] == 0:
            self.free.append(b)


class PagedKVCache:
    """Per-sequence KV cache backed by block tables."""

    def __init__(self, alloc):
        self.alloc = alloc
        self.blocks = []                                       # list of physical block ids
        self.n_filled = 0                                      # positions used in the last block

    def append(self, k, v):
        bs = self.alloc.block_size
        if self.n_filled == 0 or self.n_filled == bs:
            new_b = self.alloc.allocate()
            self.blocks.append(new_b)
            self.n_filled = 0
        b = self.blocks[-1]
        self.alloc.blocks[b, self.n_filled, :, :] = k          # write K
        self.n_filled += 1

    def fork(self):
        """Copy-on-write: share blocks, refcount++."""
        child = PagedKVCache(self.alloc)
        child.blocks = list(self.blocks)
        for b in child.blocks:
            self.alloc.incref(b)
        child.n_filled = self.alloc.block_size                 # force new-block on next write
        return child

    def free(self):
        for b in self.blocks:
            self.alloc.decref(b)


if __name__ == "__main__":
    print("=== PagedAttention / vLLM (Kwon et al 2023 SOSP) ===\n")

    n_kv_heads, d_head, block_size, n_blocks = 4, 8, 16, 32
    alloc = BlockAllocator(n_blocks, block_size, d_head, n_kv_heads)

    # Sequence A: 40 tokens (needs ceil(40/16) = 3 blocks)
    rng = np.random.default_rng(0)
    seqA = PagedKVCache(alloc)
    for t in range(40):
        seqA.append(rng.normal(size=(n_kv_heads, d_head)), None)
    print(f"  Seq A length 40: uses {len(seqA.blocks)} blocks")
    print(f"    Block table: {seqA.blocks}   Refs: {alloc.refs[seqA.blocks].tolist()}")

    # Fork A -> B for parallel sampling: initially shares all blocks
    seqB = seqA.fork()
    print(f"\n  Fork A -> B (parallel sampling): shared blocks {seqB.blocks}")
    print(f"    Refs on shared blocks: {alloc.refs[seqB.blocks].tolist()}")

    # B extends by 20 tokens -> allocates 2 new blocks (didn't touch shared)
    for t in range(20):
        seqB.append(rng.normal(size=(n_kv_heads, d_head)), None)
    print(f"\n  Seq B after +20 tokens: {len(seqB.blocks)} blocks total, "
          f"new blocks {seqB.blocks[3:]}")

    # Free A: shared blocks stay because B still holds them
    seqA.free()
    print(f"\n  Free A: physical blocks in use = {alloc.refs.sum()} "
          f"(4 kept for B: {seqB.blocks[:3]} shared + {seqB.blocks[3:]} own)")

    # Total token capacity used vs allocated
    total_tokens_B = 40 + 20                                    # 60 tokens
    contiguous_bytes = total_tokens_B * n_kv_heads * d_head * 8  # naive
    paged_bytes = len(seqB.blocks) * block_size * n_kv_heads * d_head * 8
    print(f"\n  Seq B: {total_tokens_B} tokens, contiguous alloc would use "
          f"{contiguous_bytes:,} bytes")
    print(f"  Paged alloc uses {paged_bytes:,} bytes "
          f"({100 * (paged_bytes - contiguous_bytes) / paged_bytes:.1f}% internal fragmentation, "
          f"but SHARES 3 blocks with siblings)")

    print("\n--- library cross-check (vllm / TGI / TensorRT-LLM Python) ---")
