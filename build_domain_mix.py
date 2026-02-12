"""
Build a mixed-domain dataset for MoE experiments.

Creates a single training file mixing three domains:
  1. Names (from names.txt — makemore's default dataset)
  2. Arithmetic expressions (e.g., "23+45=68")
  3. Tiny Python snippets (e.g., "x=x+1")

Each line is one example. We also create per-domain validation sets
for analyzing whether MoE routing correlates with domain.

Usage:
    python build_domain_mix.py                  # default (~96K examples)
    python build_domain_mix.py --scale 0.25     # small (~24K, for quick tests)
    python build_domain_mix.py --scale 1.0      # full (~96K)

Outputs:
    data/domain_mix_train.txt   — mixed training data
    data/domain_mix_test.txt    — mixed test data
    data/domain_names_val.txt   — names-only validation set
    data/domain_arith_val.txt   — arithmetic-only validation set
    data/domain_code_val.txt    — code-only validation set
    data/domain_labels.txt      — per-line domain labels for the training set
"""

import argparse
import os
import random

random.seed(42)

# ---------------------------------------------------------------------------
# Domain 1: Names
# ---------------------------------------------------------------------------

def load_names(path="names.txt", n_train=None, n_val=500):
    with open(path) as f:
        names = [w.strip().lower() for w in f if w.strip()]
    random.shuffle(names)
    if n_train is None:
        n_train = len(names) - n_val  # use all available names
    return names[:n_train], names[n_train:n_train + n_val]

# ---------------------------------------------------------------------------
# Domain 2: Arithmetic expressions
# ---------------------------------------------------------------------------

def generate_arithmetic(n_train=32000, n_val=500):
    """Generate simple arithmetic strings like '23+45=68' and '7*6=42'.

    Uses wider operand ranges to ensure enough unique examples.
    """
    examples = set()
    attempts = 0
    max_attempts = (n_train + n_val) * 20
    while len(examples) < n_train + n_val and attempts < max_attempts:
        attempts += 1
        op = random.choice(['+', '-', '*'])
        if op == '+':
            a = random.randint(0, 999)
            b = random.randint(0, 999)
            result = a + b
        elif op == '-':
            a = random.randint(0, 999)
            b = random.randint(0, a)  # keep result non-negative
            result = a - b
        elif op == '*':
            a = random.randint(0, 99)
            b = random.randint(0, 99)
            result = a * b
        expr = f"{a}{op}{b}={result}"
        examples.add(expr)
    if len(examples) < n_train + n_val:
        print(f"  Warning: only generated {len(examples)} unique arithmetic expressions "
              f"(requested {n_train + n_val})")
    examples = list(examples)
    random.shuffle(examples)
    return examples[:n_train], examples[n_train:n_train + n_val]

# ---------------------------------------------------------------------------
# Domain 3: Tiny code snippets
# ---------------------------------------------------------------------------

CODE_TEMPLATES = [
    # Variable assignments
    lambda: f"{rv()}={random.randint(0,99)}",
    lambda: f"{rv()}={rv()}+{random.randint(1,9)}",
    lambda: f"{rv()}={rv()}-{random.randint(1,9)}",
    lambda: f"{rv()}={rv()}*{random.randint(2,9)}",
    # Simple if statements
    lambda: f"if {rv()}>{random.randint(0,50)}:{rv()}={random.randint(0,9)}",
    lambda: f"if {rv()}=={random.randint(0,9)}:{rv()}={rv()}+1",
    # For loops
    lambda: f"for {rv()} in range({random.randint(2,20)}):{rv()}={rv()}+1",
    lambda: f"for {rv()} in range({random.randint(2,10)}):{rv()}={rv()}*2",
    # Function-like
    lambda: f"def {rf()}({rv()}):{rf2()}({rv()})",
    lambda: f"return {rv()}+{rv()}",
    lambda: f"print({rv()})",
    # List operations
    lambda: f"{rv()}=[{random.randint(0,9)},{random.randint(0,9)},{random.randint(0,9)}]",
    lambda: f"{rv()}=len({rv()})",
    lambda: f"{rv()}.append({random.randint(0,9)})",
]

_VARS = list("xyznabc")
_FUNCS = ["foo", "bar", "add", "run", "get", "put", "calc"]
_FUNCS2 = ["print", "len", "abs", "sum", "max", "min"]

def rv():
    return random.choice(_VARS)

def rf():
    return random.choice(_FUNCS)

def rf2():
    return random.choice(_FUNCS2)

def generate_code(n_train=32000, n_val=500):
    examples = set()
    attempts = 0
    max_attempts = (n_train + n_val) * 10
    while len(examples) < n_train + n_val and attempts < max_attempts:
        attempts += 1
        template = random.choice(CODE_TEMPLATES)
        examples.add(template())
    examples = list(examples)
    random.shuffle(examples)
    return examples[:n_train], examples[n_train:n_train + n_val]

# ---------------------------------------------------------------------------
# Build the mixed dataset
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor for dataset size (1.0 = full, 0.25 = quick test)")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    # Names: use all available (up to ~32K), scaled
    names_train, names_val = load_names(n_train=int(31000 * args.scale), n_val=500)
    n_target = len(names_train)  # match other domains to names count

    arith_train, arith_val = generate_arithmetic(n_train=n_target, n_val=500)
    code_train, code_val = generate_code(n_train=n_target, n_val=500)

    print(f"Names:      {len(names_train)} train, {len(names_val)} val")
    print(f"Arithmetic: {len(arith_train)} train, {len(arith_val)} val")
    print(f"Code:       {len(code_train)} train, {len(code_val)} val")

    # Combine and shuffle training data, keeping track of domain labels
    train_items = (
        [(w, "name") for w in names_train] +
        [(w, "arith") for w in arith_train] +
        [(w, "code") for w in code_train]
    )
    random.shuffle(train_items)
    train_words = [w for w, _ in train_items]
    train_labels = [l for _, l in train_items]

    # Combine test data similarly
    test_items = (
        [(w, "name") for w in names_val] +
        [(w, "arith") for w in arith_val] +
        [(w, "code") for w in code_val]
    )
    random.shuffle(test_items)
    test_words = [w for w, _ in test_items]

    # Write outputs
    with open("data/domain_mix_train.txt", "w") as f:
        f.write("\n".join(train_words) + "\n")
    with open("data/domain_mix_test.txt", "w") as f:
        f.write("\n".join(test_words) + "\n")
    with open("data/domain_labels.txt", "w") as f:
        f.write("\n".join(train_labels) + "\n")

    # Per-domain validation sets (for routing analysis)
    with open("data/domain_names_val.txt", "w") as f:
        f.write("\n".join(names_val) + "\n")
    with open("data/domain_arith_val.txt", "w") as f:
        f.write("\n".join(arith_val) + "\n")
    with open("data/domain_code_val.txt", "w") as f:
        f.write("\n".join(code_val) + "\n")

    # Summary stats
    all_chars = sorted(set("".join(train_words)))
    max_len = max(len(w) for w in train_words)
    avg_lens = {}
    for label in ["name", "arith", "code"]:
        items = [w for w, l in zip(train_words, train_labels) if l == label]
        avg_lens[label] = sum(len(w) for w in items) / len(items) if items else 0

    print(f"\nMixed dataset: {len(train_words)} train, {len(test_words)} test")
    print(f"Vocabulary: {len(all_chars)} chars: {''.join(all_chars)}")
    print(f"Max length: {max_len}")
    print(f"Avg lengths: name={avg_lens['name']:.1f}, arith={avg_lens['arith']:.1f}, code={avg_lens['code']:.1f}")
    print(f"\nSample names: {names_train[:3]}")
    print(f"Sample arith: {arith_train[:3]}")
    print(f"Sample code:  {code_train[:3]}")

if __name__ == "__main__":
    main()
