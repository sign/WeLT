# Benchmarks

All numbers are on a single NVIDIA GB10 (DGX Spark: unified LPDDR5x memory, ~95 TFLOP/s measured bf16 matmul peak).
The GB10 is memory-bandwidth bound, so on e.g. H100s the relative gains of each optimization may differ.

## Time per training step

### Bench config: HuggingFace Trainer vs. Megatron-Bridge

Same model sizes and data in both stacks:
`Helsinki-NLP/opus-100` en-he, packed to 128 words per example, 32 bytes per word, micro batch 32, bf16,
a 10-layer 256-wide bytes encoder, a 6-layer 512-wide latent transformer, and a `sbintuitions/tiny-lm`-shaped
bytes decoder, from scratch. Time per step is the steady-state mean (after warmup).

| Stack | ms / step | Speedup | Model TFLOP/s |
|-------|----------:|--------:|--------------:|
| HF Trainer (`huggingface-transformers` tag) | 1227 | 1.0x | 2.3 |
| Megatron-Bridge ([`welt-bench.yaml`](welt-bench.yaml)) | 127 | **9.7x** | 21.8 |

Both learn comparably: after 300 steps the per-byte loss is 0.92 for HF (mean of steps 251-300, linear LR decay)
and 0.86 for Megatron-Bridge (mean of steps 291-300, cosine LR decay).
The HF baseline encodes bytes with a BERT encoder, Megatron-Bridge with a Llama-like bidirectional encoder of the same size.

Reproduce with `torchrun --nproc_per_node=1 -m welt_training.train benchmarks/welt-bench.yaml`. The HF Trainer
harness, and the time per step of every optimization attempt, are in git history (commit bc057a5).

### What made it faster

1. **Megatron-Bridge** (2.0x): Transformer Engine layers with fused RoPE / SwiGLU / RMSNorm / cross entropy,
   distributed fused Adam, no HF Trainer overheads.
2. **Packed bytes** (2.4x): the bytes of all words are packed (THD) for the encoders and the decoder, instead of
   padding every word to the longest one in the batch (words average ~7 bytes; padded to 24-32).
3. **FlexAttention** (1.5x on MT): varlen flash attention is slow for tens of thousands of few-token sequences
   (its backward pads each sequence to a 128 tile). FlexAttention with a block mask built in O(tokens) from
   `cu_seqlens` is ~2x faster per layer, and also replaces Transformer Engine's unfused attention for the latent
   transformer's arbitrary (packed + bidirectional shift block) mask.
4. **Encode unique words** (1.3x on MT): the word encoders' output only depends on the word, and only ~31% of the
   words in a batch are distinct, so each distinct word is encoded once and its embedding is shared (exact).
5. **Packed patches** (SignWriting 404 -> 262 ms): the image encoder packs each batch's patches, with patch row and
   column embeddings, instead of padding every sign (28 patches on average) to the batch's largest (~105).

Not kept: padded SDPA / matmul attention for the short sequences (memory-bound on GB10), FP8 (MXFP8 is unsupported on
SM12x; tensorwise FP8 needs token counts padded to multiples of 8, and matmuls are a minority of the step time here).

### Utilization

[`flops.py`](flops.py) counts the FLOPs of each transformer on its real (packed) tokens: "model" FLOPs encode every word
occurrence, "hardware" FLOPs encode each distinct word once
(`python benchmarks/flops.py <config.yaml> --ms_per_step <ms>`).

| Config | ms / step | Model TFLOPs / step | Model TFLOP/s | Hardware TFLOP/s |
|--------|----------:|-------------------:|--------------:|-----------------:|
| Bench | 127 | 2.76 | 21.8 | 14.8 |
| Machine translation (Muon, task run) | 192 | 3.04 | 15.8 | 12.9 |

The remaining time is dominated by memory-bound kernels (SwiGLU, RoPE, norms) of the narrow (128-512 wide)
transformers, which GB10's bandwidth limits.

### Profile: SignWriting, image + bytes encoders

Measured on GB10: 95 TFLOP/s bf16 matmul (8192³), but ~35 TFLOP/s for the 256-wide layers' matmuls (memory bound),
223 GB/s device memory, 57 GB/s host to device. A step of
[`machine-translation-signed-spoken.yaml`](../welt_training/experiments/machine-translation/machine-translation-signed-spoken.yaml)
(96 examples of 64 words: 6144 words, 1935 distinct) takes 246 ms (279 before the changes below):

| Part | ms | Notes |
|------|---:|-------|
| Bytes encoder, forward + backward | 114 | 86k packed bytes (44.7 per sign): memory bound (matmuls, SwiGLU, RoPE, norms at bandwidth) |
| Image encoder | 23 | 28.7k patches (with CLS) |
| Latent transformer | 29 | |
| Bytes decoder (with LM head) | 27 | |
| Optimizer (Muon) | 33 | 21.6 ms of kernels, the rest launching ~3900 kernels |
| GPU idle | ~18 | the CPU launching kernels (~5700 per step) after host synchronizations |

- **Data**: 4 workers render a batch in ~75-100 ms, well within a step. A batch is 51 MB (44.5 MB of uint8 patches,
  padded to the example with the most) and its copy takes 0.9 ms, but copied from the workers' shared memory it page
  faults (~10 ms), and each new batch shape allocates (~4 ms): batches are now pinned in the data loader's thread,
  off the step (-12 ms). Fewer bytes (unique words' patches only, uint8 ids) would not shorten the step any more.
- **Embeddings of valid bytes only** (-21 ms): the encoder and decoder embedded every padded byte position, then
  gathered the valid ones with a boolean mask, whose backward sorts indices. They now gather the bytes, then embed.
- **Attention**: FlexAttention over the packed words (6.8 ms per layer forward + backward for the bytes encoder) is
  faster than FlashAttention 2 varlen (7.9) and PyTorch's varlen attention (9.2); cuDNN's THD backward is
  unsupported on SM12x.
- Left: the unique-words sort and boolean masks synchronize the host each step (~8 ms of sorting); the optimizer's
  launches. Beyond that, the bytes encoder is bound by memory bandwidth over its packed bytes: fewer bytes per sign
  (SignWriting's 4-byte characters) is a design change, not an optimization.

Profile any config with `profiling.use_pytorch_profiler=true profiling.profile_step_start=60
profiling.profile_step_end=64` (a trace in `<output_dir>/torch_profile`) and `logger.timing_log_level=2`.

### Multiple GPUs

[`parity.sh`](parity.sh) trains the machine translation config for 200 steps on the same global batches
(64 examples, micro batch 32), on 2 H100s (on Modal, 16 CPUs):

| Parallelism | ms / step | Val. loss after 200 steps |
|-------------|----------:|--------------------------:|
| 1 GPU (2 micro batches) | 225 | 0.935 |
| Data parallel (DP=2) | 122 | 0.929 |

Data parallelism scales 1.8x (median step times; Modal hosts vary).

With micro batch 64, a step takes 103 ms on one H100 vs. 167 ms on the GB10: only 1.6x faster. A profile shows the
H100 is host bound: its kernels take ~30 ms of a ~125 ms step, the rest is the CPU launching ~1500 small kernels
(Transformer Engine modules cost 0.2-0.4 ms of CPU each for these narrow layers) and building the packed attention
masks. So on H100s, use larger micro batches (128: 1.7x the samples per second, 256: ~2.9x).
Follow-up: computing the packing metadata (lengths, unique words, block masks) in the dataloader and padding packed
shapes to fixed buckets would remove the host synchronizations and allow CUDA graphs.

## Tasks

[`run_task.sh`](run_task.sh) trains a config, exports it, and evaluates generation with vLLM
([`welt_training/evaluate.py`](../welt_training/evaluate.py): exact match and chrF of completions generated from the
validation prefixes), recording a row of [`tasks.csv`](tasks.csv).

| Task | Steps | ms / step | Train time | Val. bits / byte | Val. word acc. | Gen. exact match | Gen. chrF | Gen. words / s |
|------|------:|----------:|-----------:|-----------------:|---------------:|-----------------:|----------:|---------------:|
| **string-repetition**: repeat an English sentence, pretrained tiny LMs | 1500 | 59 | 1.5 min | 0.008 | 99.3% | 93.8% | 99.2 | 548 |
| **ocr**: write a sentence seen only as rendered word images | 3000 | 62 | 3.1 min | 0.009 | 98.8% | 87.5% | 97.8 | 449 |
| **letter-count**: count the letters of a word | 3000 | 65 | 3.2 min | 0.0001 | 100.0% | 100.0% | 100.0 | 1373 |
| **machine-translation**: English to Hebrew, from scratch, image + bytes encoders | 10000 | 182 | 30.3 min | 0.531 | 68.5% | 9.8% | 47.6 | 910 |
| **machine-translation**, bytes encoder only | 10000 | 160 | 26.8 min | 0.528 | 68.0% | 6.2% | 46.2 | 867 |
| **signed-to-spoken**: SignWriting to text ([signbank-plus](https://huggingface.co/datasets/sign/signbank-plus)), from scratch, bytes encoder | 10000 | 209 | 34.9 min | 1.244 | 63.1% | 2.7% | 10.4 | 560 |
| **signed-to-spoken**, bytes + image encoders (rendered SignWriting) | 10000 | 248 | 41.4 min | 1.246 | 63.4% | 2.3% | 12.2 | 638 |
| **signed-to-spoken**, image encoder only | 10000 | 150 | 24.9 min | 1.277 | 63.1% | 2.3% | 10.8 | 790 |

All tasks train with Muon (their configs). Steps are `train.train_iters` overrides of the configs' 10000
(string-repetition: 1500; ocr, letter-count: 3000). Generation is greedy, on 256 validation
examples (the easy tasks' configs hold out 32), with vLLM (batched over all examples). Reproduce a row with e.g.
`WANDB_MODE=disabled benchmarks/run_task.sh welt_training/experiments/easy-tasks/string-repetition.yaml
output/string-repetition train.train_iters=1500`.

Compared to the same tasks trained with Adam (before the review fixes), Muon improves every task: e.g. ocr from
80.5% to 88.3% exact match, machine translation from 0.626 to 0.542 bits per byte (chrF 41.0 to 46.2), at 10-30%
more time per step (its orthogonalization).

SignWriting to text is far from solved at this scale: the models generate fluent words of the target language,
but rarely the right one (see the examples below; vLLM matches the trained Megatron models on these prompts). The
dataset is small (57k packed examples): every configuration's validation loss is best at ~6000 steps (10 epochs) and
then overfits. Dropout in the latent transformer slows this: `hidden_dropout: 0.2` (the config's) gives the best last
step (1.270 bits per byte with both encoders, vs. 1.303 with 0.1 and 1.276 with 0.3; weight decay 0.1 did not help).

The image encoder (over SignWriting rendered by pixel-renderer, from scratch) barely helps: at their best checkpoints,
both encoders reach 1.174 bits per byte (step 7000) vs. 1.178 with bytes only (step 6000) and 1.258 with the image
encoder only. By the last step all have overfitted (bytes only is lowest there, 1.244 vs. 1.246); both encoders
generate with the best chrF (12.2), bytes only with the most exact matches (2.7%), but generation, on 256 examples, is
too noisy to rank these (chrF moves by up to 5 between checkpoints).
There is no modality dropout (one encoder's embeddings dropped at random): it was worse than none (1.25 vs. 1.23 bits
per byte, and under half the exact matches).

Images do not help machine translation: without the image encoder (`model.image_encoder=null`), English to Hebrew
reaches 0.528 bits per byte and chrF 46.2 (vs. 0.531 and 47.6 with it, within the noise of generation), 12% faster
per step.

Each task trains its own models and data (see its config): string-repetition, ocr and letter-count use tiny
transformers and short sequences, so their steps are faster than the bench config's 127 ms, while machine
translation trains a 70m latent transformer with both encoders.

The causal LM [baseline](../welt_training/experiments/machine-translation/baseline.yaml) (the same 6-layer 512-wide
transformer over Pythia BPE tokens, the same data, batch size, steps and Muon) scores the same bytes as WeLT (not the
source sentences in shift blocks): **0.599** validation bits per byte at 633 ms / step, vs. **0.542** for WeLT at
192 ms / step.

Training is bit-reproducible (the same seed gives the same losses); between seeds, machine translation's last-step
bits per byte vary by ~0.01 (0.540 and 0.531 with seeds 42 and 1234), so smaller differences are noise.
The bytes decoder's input starts with the latent in place of BOS (machine translation, image + bytes encoders, two
seeds, last step / best step):

| Bytes decoder input | Val. bits / byte | Best | ms / step |
|---|--:|--:|--:|
| latent, BOS, bytes (before) | 0.540, 0.531 | 0.533, 0.525 | 186 |
| latent in place of BOS, bytes (main) | 0.529, 0.533 | 0.527, 0.528 | 175 |

Dropping BOS is 6% faster at the same quality.

### Examples

The first 5 validation examples of each task (`␎` and `␏` stand for the shift block controls `\x0E` and `\x0F`).

**string-repetition**

| Input | Expected | Generated |
|---|---|---|
| `<text>␎Wouldn't it be more cruel for society to let people die... - ... when with some effort it could save them?␏<repeat> ` | `Wouldn't it be more cruel for society to let people die... - ... when with some effort it could save them?` | `Wouldn't it be more cruel for society to let people die... - ... when with some effort it could save them?` |
| `<text>␎YOu dOn't know the half Of it.␏<repeat> ` | `YOu dOn't know the half Of it.` | `You don't know the half Of it.` |
| `<text>␎Superman's exact opposite who lives in the backwards Bizarro World.␏<repeat> ` | `Superman's exact opposite who lives in the backwards Bizarro World.` | `Superman's exact opposite who lives in the backwards Bizarro World.` |
| `<text>␎- The apology, so- - We're keeping the robot.␏<repeat> ` | `- The apology, so- - We're keeping the robot.` | `- The apology, so- - We're keeping the robot.` |
| `<text>␎Of course, there were always mama's boyfriends, but as soon as I'd learn their names, mama would kick them out, and there'd be a new toothbrush in the bathroom.␏<repeat> ` | `Of course, there were always mama's boyfriends, but as soon as I'd learn their names, mama would kick them out, and there'd be a new toothbrush in the bathroom.` | `Of course, there were always mama's boyfriends, but as soon as I'd learn their names, mama would kick them out, and there'd be a new toothbrush in the bathroom.` |

**ocr (the model sees only the rendered words)**

| Input | Expected | Generated |
|---|---|---|
| `<text>␎Wouldn't it be more cruel for society to let people die... - ... when with some effort it could save them?␏<repeat> ` | `Wouldn't it be more cruel for society to let people die... - ... when with some effort it could save them?` | `Wouldn't it be more cruel for society to let people die... - ... when with some effort it could save them?` |
| `<text>␎YOu dOn't know the half Of it.␏<repeat> ` | `YOu dOn't know the half Of it.` | `YOu daNt! know the half Of it.` |
| `<text>␎Superman's exact opposite who lives in the backwards Bizarro World.␏<repeat> ` | `Superman's exact opposite who lives in the backwards Bizarro World.` | `Superman's exact opposite who lives in the backwards Bizarro World.` |
| `<text>␎- The apology, so- - We're keeping the robot.␏<repeat> ` | `- The apology, so- - We're keeping the robot.` | `- The apology, so- - We're keeping the robot.` |
| `<text>␎Of course, there were always mama's boyfriends, but as soon as I'd learn their names, mama would kick them out, and there'd be a new toothbrush in the bathroom.␏<repeat> ` | `Of course, there were always mama's boyfriends, but as soon as I'd learn their names, mama would kick them out, and there'd be a new toothbrush in the bathroom.` | `Of course, there were always mama's boyfriends, but as soon as I'd learn their names, mama would kick them out, and there'd be a new toothbrush in the bathroom.` |

**letter-count**

| Input | Expected | Generated |
|---|---|---|
| `<text>␎Morel␏<count> ` | `M1 O1 R1 E1 L1` | `M1 O1 R1 E1 L1` |
| `<text>␎neurosis␏<count> ` | `N1 E1 U1 R1 O1 S2 I1` | `N1 E1 U1 R1 O1 S2 I1` |
| `<text>␎mydaleine␏<count> ` | `M1 Y1 D1 A1 L1 E2 I1 N1` | `M1 Y1 D1 A1 L1 E2 I1 N1` |
| `<text>␎forgetter␏<count> ` | `F1 O1 R2 G1 E2 T2` | `F1 O1 R2 G1 E2 T2` |
| `<text>␎fiddley␏<count> ` | `F1 I1 D2 L1 E1 Y1` | `F1 I1 D2 L1 E1 Y1` |

**machine-translation**

| Input | Expected | Generated |
|---|---|---|
| `<en>␎Wouldn't it be more cruel for society to let people die... - ... when with some effort it could save them?␏<he> ` | `האם לא יהיה זה אכזרי יותר מצד החברה, לתת לאנשים למות... כאשר עם מאמץ מה ניתן להצילם?` | `לא יהיה יותר אכזרי לחברה לתת לאנשים למות... מתי מאמץ שיכול להציל אותם?` |
| `<en>␎YOu dOn't know the half Of it.␏<he> ` | `אתה לא יודע חצי מזה.` | `אתה לא יודע את החצי ממנו.` |
| `<en>␎Superman's exact opposite who lives in the backwards Bizarro World.␏<he> ` | `ההפך הגמור של סופרמן מי שחי בעולם Bizarro אחורה.` | `סופרמן הוא הדופן המדויק שגר בעולם האחורי של ביירזורק.` |
| `<en>␎- The apology, so- - We're keeping the robot.␏<he> ` | `-ההתנצלות, אז... אנחנו שומרים את הרובוט.` | `ההתנצלות, אז... אנחנו שומרות את הרובוט.` |
| `<en>␎Of course, there were always mama's boyfriends, but as soon as I'd learn their names, mama would kick them out, and there'd be a new toothbrush in the bathroom.␏<he> ` | `כמובן שתמיד היו החברים של אמא שלי, אבל כשלמדתי את שמותיהם, אמא הייתה מעיפה אותם, והייתה מברשת שיניים חדשה באמבטיה.` | `כמובן, תמיד היו החברים של אמא, אבל ברגע שאלמד את שמם, אמא תעיף אותם החוצה, ותהיה שירה חדשה בשירותים.` |

**machine-translation, bytes**

| Input | Expected | Generated |
|---|---|---|
| `<en>␎Wouldn't it be more cruel for society to let people die... - ... when with some effort it could save them?␏<he> ` | `האם לא יהיה זה אכזרי יותר מצד החברה, לתת לאנשים למות... כאשר עם מאמץ מה ניתן להצילם?` | `זה לא יהיה יותר אכזרי לחברה לתת לאנשים למות... מתי שיש מאמץ זה יכול להציל אותם?` |
| `<en>␎YOu dOn't know the half Of it.␏<he> ` | `אתה לא יודע חצי מזה.` | `אתה לא יודע מה החלק המוזר.` |
| `<en>␎Superman's exact opposite who lives in the backwards Bizarro World.␏<he> ` | `ההפך הגמור של סופרמן מי שחי בעולם Bizarro אחורה.` | `הסופר המדויק ביותר שחי בעולם ביאזרו.` |
| `<en>␎- The apology, so- - We're keeping the robot.␏<he> ` | `-ההתנצלות, אז... אנחנו שומרים את הרובוט.` | `-אנחנו שומרים את הרובוט.` |
| `<en>␎Of course, there were always mama's boyfriends, but as soon as I'd learn their names, mama would kick them out, and there'd be a new toothbrush in the bathroom.␏<he> ` | `כמובן שתמיד היו החברים של אמא שלי, אבל כשלמדתי את שמותיהם, אמא הייתה מעיפה אותם, והייתה מברשת שיניים חדשה באמבטיה.` | `כמובן שהיו חברים של אמא, אבל ברגע שאלמד את שמותיהם, אמא הייתה מכסחת שיניים והיא הייתה מחסום שיניים חדשים בחדר האמבטיה.` |

**signed-to-spoken, bytes**

| Input | Expected | Generated |
|---|---|---|
| `<ncs>␎𝠀񌀅񆊱񂌳𝠃𝤠𝥇񌀅𝣴𝣵񆊱𝣶𝤜񂌳𝤅𝤴␏<es> ` | `difficult` | `comer` |
| `<ncs>␎𝠀񌀅񆊱񂌳𝠃𝤠𝥇񌀅𝣴𝣵񆊱𝣶𝤜񂌳𝤅𝤴␏<es> ` | `dificil` | `comer` |
| `<bzs>␎𝠀񍝁񆇡񄼱񉸒𝠃𝥊𝤳񍝁𝣴𝣵񆇡𝤋𝤅񄼱𝤙𝣼񉸒𝤱𝤚␏<pt> ` | `Flavia` | `Marcos Barreto` |
| `<ssp>␎𝠀񅯱񅯵񈪇񋾡𝠃𝤨𝥅񅯱𝤙𝤗񅯵𝤘𝤪񈪇𝣽𝤞񋾡𝣴𝣴␏<es> ` | `abril` | `caracol` |
| `<bzs>␎𝠀񆀡𝠃𝤎𝤏񆀡𝣿𝣽␏<pt> ` | `M` | `n` |

**signed-to-spoken, bytes + image**

| Input | Expected | Generated |
|---|---|---|
| `<ncs>␎𝠀񌀅񆊱񂌳𝠃𝤠𝥇񌀅𝣴𝣵񆊱𝣶𝤜񂌳𝤅𝤴␏<es> ` | `difficult` | `Andrés` |
| `<ncs>␎𝠀񌀅񆊱񂌳𝠃𝤠𝥇񌀅𝣴𝣵񆊱𝣶𝤜񂌳𝤅𝤴␏<es> ` | `dificil` | `Andrés` |
| `<bzs>␎𝠀񍝁񆇡񄼱񉸒𝠃𝥊𝤳񍝁𝣴𝣵񆇡𝤋𝤅񄼱𝤙𝣼񉸒𝤱𝤚␏<pt> ` | `Flavia` | `Fernando` |
| `<ssp>␎𝠀񅯱񅯵񈪇񋾡𝠃𝤨𝥅񅯱𝤙𝤗񅯵𝤘𝤪񈪇𝣽𝤞񋾡𝣴𝣴␏<es> ` | `abril` | `altura` |
| `<bzs>␎𝠀񆀡𝠃𝤎𝤏񆀡𝣿𝣽␏<pt> ` | `M` | `S` |
