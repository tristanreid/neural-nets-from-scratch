# Neural Nets from Scratch

Companion code for the [Neural Nets from Scratch](https://tristanreid.com/blog/) blog series.

All experiments run on CPU. No GPU required.

Built on top of Karpathy's [makemore](https://github.com/karpathy/makemore) — a single-file, educational character-level language model. Each post in the series adds a modification to the base transformer and runs controlled experiments to explore a different kind of conditional computation.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build the mixed-domain dataset (~93K examples: names + arithmetic + code)
python build_domain_mix.py

# Run all Part 4 experiments (4 configs × 3 seeds, ~50 min on M-series Mac)
python run_experiments.py --run part4_v2

# Or run a quick smoke test first (~2 min)
python run_experiments.py --run test --max-steps 50
```

## Part 4: Building a Mixture-of-Experts Model

Replaces the standard feed-forward network in each transformer block with multiple expert FFNs and a learned router.

**Blog post:** [Building a Mixture-of-Experts Model](https://tristanreid.com/blog/neural-nets-mixture-of-experts/)

### Files

| File | Description |
|------|-------------|
| `makemore.py` | Karpathy's original makemore (base transformer, dataset utilities) |
| `makemore_moe.py` | MoE extension — ExpertFFN, MoEFFN (router + gating), MoETransformer |
| `build_domain_mix.py` | Generates the mixed-domain dataset (names + arithmetic + code) |
| `run_experiments.py` | Experiment runner — orchestrates training, analysis, and reporting |
| `names.txt` | ~32K baby names (makemore's default dataset) |

### Experiments

Four configurations, each trained with 3 random seeds for 20,000 steps:

1. **Dense baseline** — standard transformer FFN (~63K params)
2. **MoE top-1 with load balancing** — 4 experts, top-1 routing, aux loss=0.01 (~175K params)
3. **MoE top-1 without load balancing** — same, but aux loss=0 (demonstrates routing collapse)
4. **MoE top-2 with load balancing** — 4 experts, top-2 routing, aux loss=0.01

After training, the runner automatically:
- Analyzes per-domain routing patterns (which experts handle names vs. arithmetic vs. code)
- Computes per-domain test loss and arithmetic accuracy
- Generates sample outputs from each model

## The Series

| Part | Post | Code | Status |
|------|------|------|--------|
| 1 | [Minds, Brains and Computers](https://tristanreid.com/blog/neural-nets-origin-story/) | — | Published |
| 2 | [Neural Nets Are Simpler Than You Think](https://tristanreid.com/blog/neural-nets-simpler-than-you-think/) | — | Published |
| 3 | A Tour of Karpathy's Tutorials | — | Draft |
| 4 | Building a Mixture-of-Experts Model | `makemore_moe.py` | Draft |
| 5 | Adaptive Computation | Coming soon | — |

## License

MIT — see [LICENSE](LICENSE). Base code from Karpathy's makemore (MIT, 2022).
