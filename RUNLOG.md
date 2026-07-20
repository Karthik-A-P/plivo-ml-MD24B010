# Plivo ML

## Test 1

### Run A: Byte-level tokenizer (vocab 256) + AdamW/warmup/cosine/clip/tied-weights/scaled-init
Hypothesis: fixing optimizer/schedule/init alone would meaningfully beat
the mediocre baseline even without tokenizer changes.
Dev bpb: 2.7049 (n_params=1,298,880, steps=2000)

### Run B: Same optimizer/init fixes + BPE tokenizer (vocab 1500, trained on 2MB corpus slice)
Hypothesis: byte-level tokenization wastes context/step budget on
multi-byte Devanagari characters; BPE compression should let block_size=128
cover proportionally more real content per step.
Corpus: 7,318,592 bytes -> 2,534,942 tokens (2.9x compression vs byte-level's 7,318,592)
Dev bpb: 2.1895 (n_params=1,497,920, steps=2000)
Conclusion: CONFIRMED. BPE improved bpb by 19% (2.7049 -> 2.1895) over an
identical optimizer/schedule/init setup. Same-file token count evidence
(dev_eval.txt: 159,225 byte-tokens vs 55,641 BPE-tokens) directly supports
the "wasted context on Hindi" hypothesis.

### Run C: BPE tokenizer, vocab=700 (test: does smaller vocab converge faster in fixed steps?)
Hypothesis: fewer embeddings to learn (700 vs 1500) might converge better
within the 2000-step cap, offsetting some compression loss.
Corpus: 7,318,592 bytes -> 3,024,976 tokens (2.4x compression, vs 2.9x at vocab 1500)
Dev bpb: 2.2616 (n_params=1,369,920, steps=2000)
Conclusion: REJECTED. Vocab 700 underperformed vocab 1500 (2.2616 vs 2.1895)
despite fewer embeddings to learn. The extra compression from a larger
vocab (fewer tokens per block -> more real content per fixed context
window) outweighed the added learning difficulty of more output classes.
Convergence wasn't the bottleneck in this range; context coverage was.
Decision: keep vocab=1500 as final config.

### Run D: Muon optimizer (2D weights) + AdamW (embeddings/norms) hybrid, BPE vocab=700
Hypothesis: Muon's Newton-Schulz orthogonalization of momentum should
improve sample efficiency on 2D hidden weights within the fixed 2000-step
budget, based on community reports of ~35% step reduction vs AdamW on
nanoGPT-style speedruns.
Changed: train.py optimizer (single AdamW -> Muon for 2D matrices, AdamW
for embeddings/norms/biases), momentum=0.95, Muon lr=0.02 scaled by same
warmup/cosine shape as AdamW schedule. Same BPE vocab=700 tokenizer as
Run C for a clean comparison.
Dev bpb: 2.2616 (AdamW, vocab 700) -> 2.1299 (Muon hybrid, vocab 700)
Loss curve still decreasing steeply at step 2000 (3.59->3.53 last 100
steps), unlike prior AdamW runs which had largely flattened by step 1500.
Conclusion: CONFIRMED, and cheaply: Muon closed most of the gap to the
vocab=1500 AdamW result (2.1895) using a *smaller* vocab (700), suggesting
the optimizer swap is more parameter-efficient than scaling tokenizer
vocab. Given the still-descending loss curve, Muon likely has more
headroom than AdamW within the same step budget.
