"""Shared pytest configuration.

Cap torch to a single CPU thread for the whole test suite. The test-scale
tensors here are tiny (d ≤ 16 lattices, hidden_dim 16 backbones), so
PyTorch's default intra-op parallelism (one OpenMP worker per core) spends
more time on fork/join synchronisation than on math: measured 2026-07-24 on
the heaviest resampling test, 12 threads = 44 s, 4 = 26 s, 2 = 23 s,
1 = 18.7 s — all-core spinning is pure heat with negative speedup at this
tensor size. Production-scale runs (Modal A100) are unaffected.
"""

import torch

torch.set_num_threads(1)
