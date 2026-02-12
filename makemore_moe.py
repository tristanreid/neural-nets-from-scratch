"""
makemore with Mixture-of-Experts — extends makemore.py with MoE-FFN blocks.

This is a minimal diff against Karpathy's makemore transformer. The only
architectural change: the FFN inside each transformer block is replaced by
K expert FFNs and a learned router (top-k gating).

Usage:
    # Dense baseline (standard transformer, no MoE)
    python makemore_moe.py -i data/domain_mix_train.txt --type transformer \
        --max-steps 5000 --device cpu --num-workers 0 -o out/dense_baseline

    # MoE transformer (4 experts, top-1 routing)
    python makemore_moe.py -i data/domain_mix_train.txt --type moe \
        --num-experts 4 --top-k 1 --max-steps 5000 --device cpu \
        --num-workers 0 -o out/moe_4exp

    # Analyze routing patterns per domain
    python makemore_moe.py -i data/domain_mix_train.txt --type moe \
        --num-experts 4 --top-k 1 -o out/moe_4exp --analyze-routing \
        --resume --device cpu --num-workers 0
"""

import os
import sys
import time
import math
import json
import argparse
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

# ---------------------------------------------------------------------------
# Import base components from makemore
# ---------------------------------------------------------------------------
from makemore import (
    ModelConfig, NewGELU, CausalSelfAttention, Transformer,
    Bigram, MLP, BoW, RNN,
    CharDataset, InfiniteDataLoader,
    generate, create_datasets,
)


# ---------------------------------------------------------------------------
# Local evaluate — makemore's version references a global `args.device`
# that doesn't exist when imported. This version takes device as a parameter.
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataset, device='cpu', batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train()
    return mean_loss

# ---------------------------------------------------------------------------
# MoE components
# ---------------------------------------------------------------------------

class ExpertFFN(nn.Module):
    """A single expert: same architecture as the standard transformer FFN."""

    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.act = NewGELU()

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class MoEFFN(nn.Module):
    """
    Mixture-of-Experts FFN layer.

    Replaces the standard FFN in a transformer block with K expert FFNs
    and a router that performs top-k gating.

    The router produces logits over experts for each token. We select the
    top-k experts, compute their outputs, and combine them with the
    softmax-normalized gating weights.

    An auxiliary load-balancing loss encourages uniform expert utilization.
    """

    def __init__(self, n_embd, num_experts=4, top_k=1, aux_loss_coef=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef

        # The experts
        self.experts = nn.ModuleList([ExpertFFN(n_embd) for _ in range(num_experts)])

        # The router: projects hidden state to expert logits
        self.router = nn.Linear(n_embd, num_experts, bias=False)

        # For logging/analysis
        self._last_routing_weights = None  # (B*T, num_experts) softmax probs
        self._last_expert_indices = None   # (B*T, top_k) selected expert ids
        self._aux_loss = None

    def forward(self, x):
        B, T, C = x.size()
        x_flat = x.view(B * T, C)  # (B*T, C)

        # Router: compute gating logits and probabilities
        router_logits = self.router(x_flat)          # (B*T, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)  # (B*T, num_experts)

        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # Normalize the selected probabilities so they sum to 1
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute expert outputs for selected experts
        # For simplicity and clarity (this is educational code), we loop over
        # experts rather than doing fancy batched dispatch
        output = torch.zeros_like(x_flat)  # (B*T, C)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]    # (B*T,)
            gate_weight = top_k_probs[:, k]     # (B*T,)

            for e in range(self.num_experts):
                mask = (expert_idx == e)  # which tokens go to expert e
                if mask.any():
                    expert_input = x_flat[mask]           # (num_tokens, C)
                    expert_output = self.experts[e](expert_input)  # (num_tokens, C)
                    output[mask] += gate_weight[mask].unsqueeze(-1) * expert_output

        # Auxiliary load-balancing loss (Switch Transformer style)
        # Encourages uniform routing by penalizing concentration
        # f_i = fraction of tokens routed to expert i
        # p_i = mean router probability for expert i
        # aux_loss = num_experts * sum(f_i * p_i)  — minimized when routing is uniform
        with torch.no_grad():
            # For top-1: each token is assigned to exactly one expert
            # For top-k: use the primary (highest-prob) expert
            primary_expert = top_k_indices[:, 0]
            tokens_per_expert = torch.zeros(self.num_experts, device=x.device)
            for e in range(self.num_experts):
                tokens_per_expert[e] = (primary_expert == e).float().sum()
            f = tokens_per_expert / (B * T)  # fraction per expert

        p = router_probs.mean(dim=0)  # mean probability per expert
        self._aux_loss = self.num_experts * (f * p).sum() * self.aux_loss_coef

        # Store for analysis
        self._last_routing_weights = router_probs.detach()
        self._last_expert_indices = top_k_indices.detach()

        return output.view(B, T, C)


class MoEBlock(nn.Module):
    """Transformer block with MoE-FFN instead of standard FFN."""

    def __init__(self, config, num_experts=4, top_k=1, aux_loss_coef=0.01):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.moe = MoEFFN(config.n_embd, num_experts, top_k, aux_loss_coef)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.moe(self.ln_2(x))
        return x


class MoETransformer(nn.Module):
    """Transformer with MoE-FFN blocks."""

    def __init__(self, config, num_experts=4, top_k=1, aux_loss_coef=0.01):
        super().__init__()
        self.block_size = config.block_size
        self.num_experts = num_experts

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([
                MoEBlock(config, num_experts, top_k, aux_loss_coef)
                for _ in range(config.n_layer)
            ]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        n_params = sum(p.numel() for p in self.transformer.parameters())
        print(f"MoE transformer: {n_params/1e6:.2f}M parameters, "
              f"{num_experts} experts, top-{top_k} routing")

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
            # Add auxiliary load-balancing loss from all MoE layers
            for block in self.transformer.h:
                if hasattr(block, 'moe') and block.moe._aux_loss is not None:
                    loss = loss + block.moe._aux_loss

        return logits, loss

    def get_routing_stats(self):
        """Collect routing statistics from all MoE layers."""
        stats = []
        for i, block in enumerate(self.transformer.h):
            if hasattr(block, 'moe'):
                moe = block.moe
                if moe._last_routing_weights is not None:
                    stats.append({
                        'layer': i,
                        'routing_probs': moe._last_routing_weights,   # (B*T, E)
                        'expert_indices': moe._last_expert_indices,    # (B*T, K)
                    })
        return stats


# ---------------------------------------------------------------------------
# Domain-specific dataset helper (reuses training vocabulary)
# ---------------------------------------------------------------------------

def create_domain_dataset(domain_file, train_dataset):
    """
    Create a dataset from a domain-specific file, reusing the training
    dataset's vocabulary (chars, stoi, itos, max_word_length).

    This is critical: if we let create_datasets() build its own vocabulary,
    the character-to-index mapping won't match the model's expectations.
    E.g., 'a'→1 in a names-only vocab vs 'a'→21 in the full mixed vocab.
    """
    with open(domain_file) as f:
        words = [w.strip() for w in f if w.strip()]
    # Use the training vocabulary's chars and max_word_length
    return CharDataset(words, train_dataset.chars, train_dataset.max_word_length)


# ---------------------------------------------------------------------------
# Routing analysis
# ---------------------------------------------------------------------------

def analyze_routing(model, train_dataset, domain_labels_file, device='cpu',
                    batch_size=32, max_batches=50):
    """
    Run the model on domain-specific data and analyze routing patterns.

    For each domain (name, arith, code), compute what fraction of tokens
    are routed to each expert — the confusion-matrix-style analysis from
    the series plan.
    """
    model.eval()

    # Load per-domain validation sets
    domains = {
        'name': 'data/domain_names_val.txt',
        'arith': 'data/domain_arith_val.txt',
        'code': 'data/domain_code_val.txt',
    }

    results = {}
    for domain_name, domain_file in domains.items():
        if not os.path.exists(domain_file):
            print(f"  Skipping {domain_name} (no file {domain_file})")
            continue

        # Create dataset reusing the training vocabulary
        domain_dataset = create_domain_dataset(domain_file, train_dataset)

        loader = DataLoader(domain_dataset, shuffle=False,
                            batch_size=batch_size, num_workers=0)

        # Accumulate routing counts per layer
        layer_counts = {}  # layer_idx -> tensor of shape (num_experts,)

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if max_batches and i >= max_batches:
                    break
                X, Y = [t.to(device) for t in batch]
                logits, _ = model(X, Y)

                # Collect routing stats
                for stat in model.get_routing_stats():
                    layer = stat['layer']
                    indices = stat['expert_indices'][:, 0]  # primary expert
                    if layer not in layer_counts:
                        layer_counts[layer] = torch.zeros(model.num_experts)
                    for e in range(model.num_experts):
                        layer_counts[layer][e] += (indices == e).sum().item()

        # Normalize to fractions
        domain_results = {}
        for layer, counts in sorted(layer_counts.items()):
            total = counts.sum().item()
            fracs = (counts / total).tolist() if total > 0 else [0] * model.num_experts
            domain_results[f"layer_{layer}"] = {
                f"expert_{e}": round(f, 4) for e, f in enumerate(fracs)
            }
        results[domain_name] = domain_results
        print(f"\n  {domain_name}:")
        for layer_key, expert_fracs in domain_results.items():
            print(f"    {layer_key}: {expert_fracs}")

    # Save results
    out_path = os.path.join(args.work_dir, "routing_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Routing analysis saved to {out_path}")

    model.train()
    return results


# ---------------------------------------------------------------------------
# Per-domain evaluation + arithmetic accuracy
# ---------------------------------------------------------------------------

def analyze_domains(model, train_dataset, device='cpu', batch_size=32, work_dir='out'):
    """
    Compute per-domain test loss and arithmetic accuracy.

    Per-domain loss shows whether MoE helps some domains more than others.
    Arithmetic accuracy measures what fraction of generated expressions
    have correct answers — a direct test of whether the model learned
    the algorithm or just the format.

    Uses the training dataset's vocabulary to ensure character-to-index
    mappings match what the model was trained on.
    """
    import re

    model.eval()

    domains = {
        'name': 'data/domain_names_val.txt',
        'arith': 'data/domain_arith_val.txt',
        'code': 'data/domain_code_val.txt',
    }

    results = {}

    # --- Per-domain loss ---
    print("\n  Per-domain test loss:")
    for domain_name, domain_file in domains.items():
        if not os.path.exists(domain_file):
            print(f"    {domain_name}: skipped (no file)")
            continue
        # Reuse training vocabulary so token indices match the model
        domain_dataset = create_domain_dataset(domain_file, train_dataset)
        loss = evaluate(model, domain_dataset, device=device, batch_size=batch_size, max_batches=50)
        results[f"{domain_name}_loss"] = round(loss, 4)
        print(f"    {domain_name}: {loss:.4f}")

    # --- Arithmetic accuracy (from generated samples) ---
    print("\n  Arithmetic accuracy (generated samples):")
    num_samples = 200
    X_init = torch.zeros(num_samples, 1, dtype=torch.long).to(device)
    steps = train_dataset.get_output_length() - 1

    with torch.no_grad():
        X_samp = generate(model, X_init, steps, do_sample=True).to('cpu')

    arith_pattern = re.compile(r'^(\d+)([+\-*])(\d+)=(\d+)$')
    total_arith = 0
    correct_arith = 0
    wrong_examples = []

    for i in range(X_samp.size(0)):
        row = X_samp[i, 1:].tolist()
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word = train_dataset.decode(row)

        m = arith_pattern.match(word)
        if m:
            total_arith += 1
            a, op, b, result = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
            if op == '+':
                expected = a + b
            elif op == '-':
                expected = a - b
            elif op == '*':
                expected = a * b
            if result == expected:
                correct_arith += 1
            else:
                if len(wrong_examples) < 5:
                    wrong_examples.append(f"{word} (expected {expected})")

    if total_arith > 0:
        accuracy = correct_arith / total_arith
        results["arith_total_generated"] = total_arith
        results["arith_correct"] = correct_arith
        results["arith_accuracy"] = round(accuracy, 4)
        print(f"    {correct_arith}/{total_arith} correct ({accuracy:.1%})")
        if wrong_examples:
            print(f"    Sample errors: {', '.join(wrong_examples)}")
    else:
        print(f"    No arithmetic samples generated in {num_samples} tries")
        results["arith_total_generated"] = 0

    # Save results
    out_path = os.path.join(work_dir, "domain_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Domain analysis saved to {out_path}")

    model.train()
    return results


# ---------------------------------------------------------------------------
# Modified print_samples that works with either model type
# ---------------------------------------------------------------------------

def print_samples(model, train_dataset, test_dataset, args, num=10):
    """Sample from model and categorize as train/test/new."""
    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
    top_k = args.top_k_sampling if args.top_k_sampling != -1 else None
    steps = train_dataset.get_output_length() - 1
    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')
    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        row = X_samp[i, 1:].tolist()
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)
        if train_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif test_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)
    print('-' * 80)
    for lst, desc in [(train_samples, 'in train'), (test_samples, 'in test'),
                      (new_samples, 'new')]:
        print(f"{len(lst)} samples that are {desc}:")
        for word in lst:
            print(word)
    print('-' * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="makemore with MoE")
    # system/input/output
    parser.add_argument('--input-file', '-i', type=str, default='data/domain_mix_train.txt')
    parser.add_argument('--work-dir', '-o', type=str, default='out', help="output working directory")
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--sample-only', action='store_true')
    parser.add_argument('--analyze-routing', action='store_true',
                        help="run routing analysis on per-domain val sets")
    parser.add_argument('--analyze-domains', action='store_true',
                        help="run per-domain loss + arithmetic accuracy")
    parser.add_argument('--num-workers', '-n', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=3407)
    # sampling
    parser.add_argument('--top-k-sampling', type=int, default=-1,
                        help="top-k for sampling, -1 means no top-k")
    # model
    parser.add_argument('--type', type=str, default='moe',
                        help="model type: transformer|moe")
    parser.add_argument('--n-layer', type=int, default=2)
    parser.add_argument('--n-head', type=int, default=4)
    parser.add_argument('--n-embd', type=int, default=48)
    parser.add_argument('--n-embd2', type=int, default=64)
    # MoE-specific
    parser.add_argument('--num-experts', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=1,
                        help="number of experts to route each token to")
    parser.add_argument('--aux-loss-coef', type=float, default=0.01,
                        help="coefficient for load-balancing auxiliary loss")
    # optimization
    parser.add_argument('--batch-size', '-b', type=int, default=32)
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4)
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01)
    args = parser.parse_args()
    print(vars(args))

    # system inits
    torch.manual_seed(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)

    # init datasets
    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    print(f"dataset determined that: {vocab_size=}, {block_size=}")

    # init model
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                         n_layer=args.n_layer, n_head=args.n_head,
                         n_embd=args.n_embd, n_embd2=args.n_embd2)

    if args.type == 'moe':
        model = MoETransformer(config, num_experts=args.num_experts,
                                top_k=args.top_k,
                                aux_loss_coef=args.aux_loss_coef)
    elif args.type == 'transformer':
        model = Transformer(config)
    else:
        raise ValueError(f"Unknown model type: {args.type}")

    model.to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total model parameters: {total_params:,}")

    if args.resume or args.sample_only or args.analyze_routing or args.analyze_domains:
        model_path = os.path.join(args.work_dir, 'model.pt')
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, weights_only=True))

    if args.sample_only:
        print_samples(model, train_dataset, test_dataset, args, num=50)
        sys.exit()

    if args.analyze_routing:
        if args.type != 'moe':
            print("ERROR: --analyze-routing requires --type moe")
            sys.exit(1)
        print("\nAnalyzing routing patterns per domain...")
        analyze_routing(model, train_dataset, 'data/domain_labels.txt',
                        device=args.device)
        sys.exit()

    if args.analyze_domains:
        print("\nAnalyzing per-domain metrics...")
        analyze_domains(model, train_dataset, device=args.device, work_dir=args.work_dir)
        sys.exit()

    # init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                   weight_decay=args.weight_decay,
                                   betas=(0.9, 0.99), eps=1e-8)

    # init dataloader
    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size,
                                       pin_memory=True, num_workers=args.num_workers)

    # training loop
    best_loss = None
    step = 0
    while True:
        t0 = time.time()

        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch]
        X, Y = batch

        logits, loss = model(X, Y)

        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        t1 = time.time()

        # logging
        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

        # evaluate
        if step > 0 and step % 500 == 0:
            train_loss = evaluate(model, train_dataset, device=args.device, batch_size=100, max_batches=10)
            test_loss = evaluate(model, test_dataset, device=args.device, batch_size=100, max_batches=10)
            print(f"step {step} train loss: {train_loss:.4f} test loss: {test_loss:.4f}")

            if best_loss is None or test_loss < best_loss:
                out_path = os.path.join(args.work_dir, "model.pt")
                print(f"test loss {test_loss:.4f} is the best so far, saving to {out_path}")
                torch.save(model.state_dict(), out_path)
                best_loss = test_loss

            # Log expert utilization for MoE models
            if args.type == 'moe':
                for stat in model.get_routing_stats():
                    indices = stat['expert_indices'][:, 0]
                    counts = torch.bincount(indices, minlength=args.num_experts).float()
                    fracs = counts / counts.sum()
                    frac_str = " ".join(f"E{e}:{f:.2f}" for e, f in enumerate(fracs.tolist()))
                    print(f"  layer {stat['layer']} routing: {frac_str}")

        # sample
        if step > 0 and step % 1000 == 0:
            print_samples(model, train_dataset, test_dataset, args, num=10)

        step += 1
        if args.max_steps >= 0 and step >= args.max_steps:
            break

    # Final save
    out_path = os.path.join(args.work_dir, "model.pt")
    torch.save(model.state_dict(), out_path)
    print(f"Training complete. Model saved to {out_path}")
