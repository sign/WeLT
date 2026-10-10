# Benchmarks

All numbers are on a single NVIDIA GB10 (DGX Spark: unified LPDDR5x memory, ~95 TFLOP/s measured bf16 matmul peak).
The GB10 is memory-bandwidth bound, so on e.g. H100s the relative gains of each optimization may differ.

## Time per training step

![Time per training step](step_time.png)

[`step_time.csv`](step_time.csv) lists every iteration, including the attempts that were not kept.
Regenerate the chart with `python benchmarks/plot.py`.

### Bench config: HuggingFace Trainer vs. Megatron-Bridge

Same model sizes and data in both stacks:
`Helsinki-NLP/opus-100` en-he, packed to 128 words per example, 32 bytes per word, micro batch 32, bf16,
a 10-layer 256-wide bytes encoder, a 6-layer 512-wide latent transformer, and a `sbintuitions/tiny-lm`-shaped
bytes decoder, from scratch. Time per step is the steady-state mean (after warmup).

| Stack | ms / step | Speedup | Model TFLOP/s |
|-------|----------:|--------:|--------------:|
| HF Trainer (`huggingface-transformers` tag, [`hf_baseline/`](hf_baseline)) | 1227 | 1.0x | 2.3 |
| Megatron-Bridge ([`welt-bench.yaml`](welt-bench.yaml)) | 127 | **9.7x** | 21.8 |

Both learn comparably: after 300 steps the per-byte loss is 0.92 for HF (mean of steps 251-300, linear LR decay)
and 0.86 for Megatron-Bridge (mean of steps 291-300, cosine LR decay).
The HF baseline encodes bytes with a BERT encoder, Megatron-Bridge with a Llama-like bidirectional encoder of the same size.

Reproduce:

```shell
# Megatron-Bridge
torchrun --nproc_per_node=1 -m welt_training.train benchmarks/welt-bench.yaml

# HF Trainer: bench_hf.py adds a timing callback to the old welt_training.trainer, so it runs in a checkout of
# the huggingface-transformers tag (set up per that checkout's README), with hf_baseline/ copied in
git worktree add ../WeLT-hf huggingface-transformers
mkdir -p ../WeLT-hf/benchmarks && cp -r benchmarks/hf_baseline ../WeLT-hf/benchmarks/
cd ../WeLT-hf && python benchmarks/hf_baseline/bench_hf.py benchmarks/hf_baseline/hf-bench.yaml
```

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

Not kept: padded SDPA / matmul attention for the short sequences (memory-bound on GB10), FP8 (MXFP8 is unsupported on
SM12x; tensorwise FP8 needs token counts padded to multiples of 8, and matmuls are a minority of the step time here).

### Utilization

[`flops.py`](flops.py) counts the FLOPs of each transformer on its real (packed) tokens: "model" FLOPs encode every word
occurrence, "hardware" FLOPs encode each distinct word once.

| Config | ms / step | Model TFLOPs / step | Model TFLOP/s | Hardware TFLOP/s |
|--------|----------:|-------------------:|--------------:|-----------------:|
| Bench | 127 | 2.76 | 21.8 | 14.8 |
| Machine translation (Muon, task run) | 192 | 3.04 | 15.8 | 12.9 |

The remaining time is dominated by memory-bound kernels (SwiGLU, RoPE, norms) of the narrow (128-512 wide)
transformers, which GB10's bandwidth limits.

### Multiple GPUs

[`parity.sh`](parity.sh) trains the machine translation config for 200 steps on the same global batches
(64 examples, micro batch 32), on 2 H100s (on Modal, 16 CPUs):

| Parallelism | ms / step | Val. loss after 200 steps |
|-------------|----------:|--------------------------:|
| 1 GPU (2 micro batches) | 225 | 0.935 |
| Data parallel (DP=2) | 122 | 0.929 |
| Tensor + sequence parallel (TP=2) | 312 | 0.938 |

Data parallelism scales 1.8x (median step times; Modal hosts vary). Tensor parallelism is slower for these narrow transformers: use it only
for models that do not fit on one GPU. Its checkpoints export (resharded) to the same vLLM format.

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
| **string-repetition**: repeat an English sentence, pretrained tiny LMs | 1500 | 66 | 1.6 min | 0.014 | 99.3% | 93.8% | 98.7 | 856 |
| **ocr**: write a sentence seen only as rendered word images | 3000 | 74 | 3.7 min | 0.014 | 99.1% | 88.3% | 97.1 | 1055 |
| **letter-count**: count the letters of a word | 3000 | 64 | 3.2 min | 0.0001 | 100% | 100% | 100 | 1378 |
| **machine-translation**: English to Hebrew, from scratch, image + bytes encoders | 10000 | 198 | 33 min | 0.547 | 67.7% | 8.2% | 45.4 | 536 |
| **machine-translation**, bytes encoder only | 10000 | 170 | 28 min | 0.537 | 68.0% | 9.0% | 46.5 | 616 |
| **signed-to-spoken**: SignWriting to text ([signbank-plus](https://huggingface.co/datasets/sign/signbank-plus)), from scratch, bytes encoder | 10000 | 224 | 37 min | 1.253 | 63.1% | 2.3% | 11.1 | 378 |
| **signed-to-spoken**, bytes + image encoders (rendered SignWriting) | 10000 | 264 | 44 min | 1.270 | 63.7% | 4.3% | 14.2 | 304 |
| **signed-to-spoken**, image encoder only | 10000 | 166 | 28 min | 1.303 | 62.9% | 0.8% | 10.5 | 267 |

All tasks train with Muon (their configs), on main after the migration. Generation is greedy, on 256 validation
examples, with vLLM (batched over all examples). Reproduce a row with e.g.
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

The image encoder (over SignWriting rendered by pixel-renderer, from scratch) helps the best checkpoint: both encoders
reach 1.212 bits per byte (step 7000) vs. 1.239 with bytes only and 1.263 with the image encoder only. By the last step
all have overfitted (bytes only is lowest there, 1.253 vs. 1.270), while both encoders generate best (4.3% exact
matches, chrF 14.2); but generation, on 256 examples, is too noisy to rank these (chrF moves by up to 5 between checkpoints).
Modality dropout (training with one encoder's embeddings dropped at random) was removed: with zeros and rescaling, or
a learned embedding standing in for the dropped encoder, it was worse than none (1.25 vs. 1.23 bits per byte, and
under half the exact matches).

An earlier comparison (bytes only 0.953 bits per byte, better than with images) was wrong: configs without images read
signs split into fragments by words-segmentation 0.0.5 from the datasets cache, which did not invalidate on the
upgrade to 0.0.6 (its version is now part of the cache fingerprint).

Images do not help machine translation: without the image encoder (`model.image_encoder=null`), English to Hebrew
reaches 0.537 bits per byte and chrF 46.5 (vs. 0.547 and 45.4 with it), 14% faster
per step. The image encoder packs each batch's patches (no padding to the largest image) and embeds patch rows and
columns: before, padding signs (28 patches on average) to the batch's largest (~105) copied 425 MB per batch to the
GPU and made a SignWriting step 404 ms; packed, 41 MB and 262 ms.

Each task trains its own models and data (see its config): string-repetition, ocr and letter-count use tiny
transformers and short sequences, so their steps are faster than the bench config's 127 ms, while machine
translation trains a 70m latent transformer with both encoders.

The causal LM [baseline](../welt_training/experiments/machine-translation/baseline.yaml) (the same 6-layer 512-wide
transformer over Pythia BPE tokens, the same data, batch size, steps and Muon) scores the same bytes as WeLT (not the
source sentences in shift blocks): **0.599** validation bits per byte at 633 ms / step, vs. **0.542** for WeLT at
192 ms / step.

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
