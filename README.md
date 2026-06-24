# Bond-RFQ Foundation Model

Pretrains a decoder-only transformer on corporate-bond **RFQ sequences** to
produce a reusable, policy-independent **client-demand embedding**, then scores
RFQ outcomes downstream. It fuses two repos:

* **`rfqsim/`** — synthetic RFQ generator for a sell-side IG corp-bond desk
  (widened to a desk-realistic universe here).
* **`src/tokenizer/`, `src/`, `scripts/`, `configs/`** — the NVIDIA
  transaction-foundation-model recipe, repurposed from card transactions to
  RFQs (CLM pretraining via NeMo AutoModel + a Llama decoder).

## Core design decisions

**Sequencing axis: per-client.** A client's time-ordered RFQ history is the
sequence (client ≈ cardholder in the original TFM). Last-token / point-in-time
pooling yields "this client's state."

**Exogenous fields only.** The embedding must be reusable across *any* desk
pricing policy, so everything endogenous to the desk is stripped from
pretraining: `our_quote`, `cover_price`, `status`, and `client_id` itself
(the model must *infer* client state from behaviour, not memorise an ID).
The 9 tokenized fields are all client/instrument/regime intrinsics:

    client_type, client_tier, sector, mat_bucket, age_bucket,
    side, size_bucket (quantile), k_dealers, mmpp_state

Vocab is a tiny **75 tokens**. `size` is the only continuous field kept and is
**quantile-binned** (32 bins) because size carries tier/type behaviour.

**Right-sized model.** The RFQ generative process has a small, known latent
dimension and a deliberate intent-noise ceiling, so the model is scaled *down*
from the 29M reference to ~**4–8M params** (a 29M model memorises). Model size
is the one open empirical knob — see the size sweep.

**The downstream task is residual.** Pretraining never sees an outcome or a
quote, so the embedding is a pure *client-demand prior*. Win-prediction is then
`[ FM client-embedding ‖ live-RFQ auction features ] -> head`, and the
**scientifically meaningful metric is the LIFT over an auction-features-only
baseline** — that delta is exactly the client-state signal, bounded above by
the simulator's intent-noise ceiling.

## Leakage control (the 2×2 partition)

Every reported number depends on three hard invariants, enforced in
`src/corpus.py` and verified on real data:

```
                 |  TRAIN weeks            |  TEST weeks (last 6)
    SEEN  (90%)  |  Cell A  pretrain+head  |  Cell B  HEADLINE eval
    HOLDOUT(10%) |  Cell C  context-only   |  Cell D  generalization eval
```

1. **Pretrain only on Cell A** → train-only pretraining + holdout-client novelty.
2. **Point-in-time embeddings**: to score RFQ *i*, pool only that client's RFQs
   strictly before *i* (causal hidden state at position *i−1*). One forward
   pass per client window yields all prefix embeddings at once.
3. **Cells B/D are scored, never trained on; Cell C is context-only.**

Headline metric = **Cell B** (seen clients, future weeks — the deployed-desk
scenario). Honesty check = **Cell D** (novel clients). If Cell B is strong but
Cell D collapses, the FM is memorising client identities, not learning state.

## Pipeline

### 1. Generate the RFQ dataset (desk-realistic universe)

```bash
# full year, 50k CUSIPs / ~1667 issuers / 10k clients, auto-detect GPUs
python rfqsim/run.py --ig-desk --out data/igdesk
```

The `--ig-desk` preset widens the *instrument* universe (CUSIPs 30k→50k) while
keeping the client axis ~10k. This was verified not to distort any correlation
structure: client↔CUSIP affinity is mediated through fixed-cardinality
`(sector, mat_bucket, age_bucket)` buckets and is invariant to bond count; the
fragmentation Gini holds/sharpens (0.865 > 0.85 target). Validation runs
automatically — gate on it before training.

`mat_bucket` and `age_bucket` were added to the generator output (they were
computed internally for client sampling but not emitted); they are exogenous
instrument features the FM consumes.

### 2. Build the leakage-safe corpus

```bash
python scripts/generate_rfq_corpus.py \
    --rfq-dir data/igdesk --out data/rfq_corpus \
    --test-weeks 6 --holdout-frac 0.10 --min-history 20 \
    --window 440 --stride 220 --size-bins 32
```

Writes `cellA_train.txt`, `cellA_val.txt`, `rfq_vocab.json`, `manifest.parquet`.
The tokenizer's size bins are fit on **train weeks only** (no test leakage into
bin edges). The manifest is the contract for the downstream extraction + eval.

### 3. Pretrain (NeMo AutoModel + Llama decoder)

```bash
python scripts/train_decoder_model.py \
    -c configs/pretrain_bond_rfq.yaml \
    --dataset.data_path data/rfq_corpus/cellA_train.txt \
    --dataset.vocab_path data/rfq_corpus/rfq_vocab.json \
    --validation_dataset.data_path data/rfq_corpus/cellA_val.txt \
    --validation_dataset.vocab_path data/rfq_corpus/rfq_vocab.json
```

Trained *past* the reference 30-step demo; tune `max_steps` against the
val-loss curve. With ~8M params and ~227M tokens/yr this is Chinchilla-fed and
runs multiple epochs.

## Status / what's validated

The full pipeline is built and **verified end-to-end on CPU** (the correctness
oracle), with GPU fast paths added for the H200 box:

Data + tokenizer + corpus + CLM-dataset + model-config: corpus build passes all
6 leakage invariants; the exact YAML model config instantiates at ~4.2M params;
step-0 loss ≈ ln(75) (correct random-init); loss descends on real Cell A data.

**Embedding extraction (leakage-critical) is proven:** the point-in-time
extractor was tested by perturbing a future RFQ and confirming embeddings for
all prior RFQs stay bit-identical while only later ones change — i.e. each
embedding depends on strictly-prior history only (Invariant 2). Context-carry
tiling preserves prefixes across window boundaries with no future leakage.

**Lift harness** (auction baseline vs `[auction ‖ embedding]` augmented,
evaluated on Cells B and D) is built; the auction baseline deliberately excludes
client_type/tier so the FM must earn the lift via learned client latents.

## GPU optimization & the parity gate

The code ships two paths: a numpy/CPU **correctness oracle** and a GPU **fast
path**. The fast paths are proven equivalent to the oracle *by construction* on
CPU; a parity test confirms them on real hardware before any full run.

What's optimized for the H200 box:

* **cuDF/cuPy tokenizer** (`src/tokenizer/rfq_tokenizer_gpu.py`): replaces the
  numpy per-element dict lookups with fully vectorised GPU ops (token id =
  precomputed field offset + integer code; size bins via `cupy.searchsorted`).
  Proven bit-identical to the numpy tokenizer across 2.7M token ids.
* **Packed token-ID corpus** (`--backend gpu`): writes the corpus as
  `(tokens.npy, offsets.npy)` int32 arrays instead of text — no string assembly,
  no load-time parse. Proven token-for-token identical to the text corpus. ~25%
  smaller and loads near-instantly vs parsing a multi-GB text file.
* **GPU-gather extractor** (`make_gpu_gather`): runs the forward in bf16
  autocast and gathers only the needed sep-positions on-GPU, copying back just
  `(n_scored, D)` instead of the full `(B, T, D)` activation tensor. Gather
  indexing proven bit-identical to the oracle on CPU.

**RUN THE PARITY TEST FIRST on the box:**

```bash
python scripts/parity_test.py --rfq-dir data/igdesk --corpus-dir data/rfq_corpus
# exit 0 = GPU paths match the oracles within tolerance; safe to scale.
```

It asserts (1) cuDF tokenizer == numpy tokenizer bit-for-bit, and (2) GPU-gather
embeddings == CPU oracle within bf16 tolerance (max abs diff < 5e-2, cosine >
0.999). A larger gap signals a bug, not precision.

## Full run sequence (H200)

```bash
# 0. parity gate
python scripts/parity_test.py --rfq-dir data/igdesk --corpus-dir data/rfq_corpus

# 1. generate desk-realistic universe (validation runs automatically)
python rfqsim/run.py --ig-desk --out data/igdesk

# 2. build packed corpus with the cuDF tokenizer
python scripts/generate_rfq_corpus.py --rfq-dir data/igdesk --out data/rfq_corpus \
    --backend gpu --test-weeks 6 --holdout-frac 0.10 --min-history 20 \
    --window 440 --stride 220 --size-bins 32

# 3. pretrain (packed config)
python scripts/train_decoder_model.py \
    -c configs/pretrain_bond_rfq_packed.yaml \
    --dataset.tokens_path data/rfq_corpus/cellA_train_tokens.npy \
    --dataset.offsets_path data/rfq_corpus/cellA_train_offsets.npy \
    --validation_dataset.tokens_path data/rfq_corpus/cellA_val_tokens.npy \
    --validation_dataset.offsets_path data/rfq_corpus/cellA_val_offsets.npy

# 4. extract point-in-time embeddings (cuDF tok + GPU gather)
python scripts/extract_embeddings_gpu.py --rfq-dir data/igdesk \
    --corpus-dir data/rfq_corpus \
    --checkpoint models/bond-rfq-fm/checkpoints/<step> \
    --out data/rfq_corpus/embeddings.parquet --gpu-tokenizer

# 5. measure lift (headline = Cell B, honesty = Cell D)
python eval/run_lift_eval.py --rfq-dir data/igdesk \
    --corpus-dir data/rfq_corpus --embeddings data/rfq_corpus/embeddings.parquet
```

### A note on the hardware / model-size mismatch

A 4–8M-param model on 75-vocab sequences fits trivially on one H200 and uses a
small fraction of its 141GB / ~990 bf16 TFLOPS — so there is no need for
multi-GPU or FSDP2 (launch with plain `python`, not `torchrun --nproc-per-node`).
The card is heavily underutilised by a single small model, which is fine: it
just means training runs are short.

Because there is one GPU, a model-size sweep runs **serially**, not in parallel.
That makes a full grid (3 hidden × 3 layers = 9 runs) less attractive on
wall-clock; prefer a small deliberate sweep along the axis that matters most
(hidden size, ~3 points, layers fixed), selected by downstream Cell-B lift. The
packed corpus + cuDF tokenizer remove the data-side bottlenecks so each run is
compute-bound and quick.

## Still to build

A serial model-size sweep (a simple loop over ~3 hidden sizes calling the
existing generate → train → extract → lift scripts, ranked by Cell-B lift) — but
only if a single sensible model (hidden=256, 6 layers) does not already show
strong lift near the simulator's intent-noise ceiling. If Cell-B lift plateaus
below that ceiling, multi-seed universe generation is the generalization lever.

Recommended order on the box: train ONE model end-to-end first, run the full
extract → lift chain, confirm real Cell-B lift, then decide whether a sweep is
worth the serial wall-clock.

## Dev-box note

`src/tokenizer/__init__.py` lazily imports the GPU (cuDF/cuML) tokenizers so the
RFQ tokenizer and corpus tooling run on a CPU dev box. The RFQ tokenizer and
corpus code are backend-neutral (numpy/pandas); the cuDF and packed paths are
the production fast paths, gated by the parity test.
