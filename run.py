"""
MacDock — Autonomous Drug Discovery Runner
===========================================
By Emmanuel MacDan

This is the self-driving loop. It uses the Claude API to propose molecular
modifications, runs real AutoDock Vina docking, and decides what to keep.
Runs forever until you stop it (Ctrl+C).

Usage:
  export ANTHROPIC_API_KEY=sk-...
  uv run run.py              # Run indefinitely
  uv run run.py --max 50     # Run 50 experiments
  uv run run.py --model claude-haiku-4-5-20251001  # Use Haiku for cheaper/faster

Every experiment is logged to results.tsv. The current best molecule is
saved to best.smi.
"""

import os
import re
import sys
import csv
import json
import time
import argparse
import traceback
from datetime import datetime
from pathlib import Path


def load_dotenv(env_path):
    """Load environment variables from a .env file (no external deps)."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


# Load .env before importing anthropic (which reads the API key at init)
load_dotenv(Path(__file__).parent / ".env")

from anthropic import Anthropic

from prepare import (
    load_config, prepare_environment, evaluate_molecule,
    validate_smiles, MoleculeResult, DOCKING_BACKEND
)

PROJECT_DIR = Path(__file__).parent
RESULTS_PATH = PROJECT_DIR / "results.tsv"
BEST_PATH = PROJECT_DIR / "best.smi"
HISTORY_PATH = PROJECT_DIR / "history.jsonl"

DEFAULT_MODEL = "claude-opus-4-7"

# ---------------------------------------------------------------------------
# System prompt: teaches the AI medicinal chemistry
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert medicinal chemist running an autonomous drug discovery loop.

Your job: propose ONE modification to the current best molecule at each iteration. The modification should improve the composite_score (lower = better).

The composite score combines:
- Binding affinity (40%): Vina docking score in kcal/mol, more negative = better. Good: < -8. Weak: > -6.
- Drug-likeness (20%): QED + Lipinski Rule of 5. Higher QED = more drug-like.
- Toxicity (20%): PAINS + Brenk structural alerts. Fewer = safer.
- Synthesizability (20%): SA score 1-10. Lower = easier to make.

You must output ONLY a JSON object with this exact format (no other text):
{
  "smiles": "your_new_SMILES_string",
  "description": "what you changed and why",
  "strategy": "bioisosteric|fragment_growing|fragment_removal|ring_modification|scaffold_hop|warhead_change|property_guided"
}

Modification strategies you can use:
- BIOISOSTERIC REPLACEMENTS: -OH ↔ -NH2, -COOH ↔ tetrazole, -F ↔ -Cl ↔ -CF3, phenyl ↔ pyridine
- FRAGMENT GROWING: add methyl, hydroxyl, fluorine, amine for H-bonds/stability
- FRAGMENT REMOVAL: simplify to improve SA score and reduce MW
- RING MODIFICATIONS: benzene → pyridine (add N), 5-ring ↔ 6-ring, fused rings
- WARHEAD CHANGES (covalent inhibitors): -C#N ↔ -C=O ↔ vinyl sulfone ↔ acrylamide
- PROPERTY-GUIDED: if QED bad → reduce MW/logP; if SA bad → simplify; if tox alerts → remove PAINS patterns

IMPORTANT RULES:
1. Output MUST be valid JSON with no surrounding text, no markdown fences.
2. SMILES MUST be valid (will be validated by RDKit).
3. Look at the history below — DO NOT repeat molecules you already tried.
4. Make SMALL changes from the current best (one modification at a time).
5. If recent attempts have been failing, try a more conservative change.
6. Keep molecular weight reasonable (200-600 Da).
"""


def parse_ai_response(text: str) -> dict:
    """Extract JSON from AI response, handling markdown fences and stray text."""
    text = text.strip()

    # Try to find JSON in markdown fences first
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        # Look for first { ... } block
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)

    return json.loads(text)


def init_results_file():
    """Create results.tsv with header if it doesn't exist."""
    if not RESULTS_PATH.exists():
        with open(RESULTS_PATH, "w") as f:
            f.write("iter\ttimestamp\tsmiles\tcomposite\tbinding\tqed\tsa\ttox\tlipinski\tmw\tlogp\tstatus\tstrategy\tdescription\n")


def log_result(iter_num, smiles, result, status, strategy, description):
    """Append a row to results.tsv."""
    timestamp = datetime.now().isoformat(timespec="seconds")
    with open(RESULTS_PATH, "a") as f:
        if result is None:
            f.write(f"{iter_num}\t{timestamp}\t{smiles}\t999.000000\t0.00\t0.0000\t0.00\t0\t0\t0.0\t0.00\t{status}\t{strategy}\t{description}\n")
        else:
            f.write(
                f"{iter_num}\t{timestamp}\t{smiles}\t"
                f"{result.composite_score:.6f}\t{result.binding_affinity:.2f}\t"
                f"{result.qed_score:.4f}\t{result.sa_score:.2f}\t{result.toxicity_alerts}\t"
                f"{result.lipinski_violations}\t{result.molecular_weight:.1f}\t{result.logp:.2f}\t"
                f"{status}\t{strategy}\t{description}\n"
            )


def save_best(smiles, result, iter_num, description):
    """Save current best molecule."""
    with open(BEST_PATH, "w") as f:
        f.write(f"{smiles}\n")
        f.write(f"# iteration: {iter_num}\n")
        f.write(f"# composite: {result.composite_score:.6f}\n")
        f.write(f"# binding: {result.binding_affinity:.2f}\n")
        f.write(f"# qed: {result.qed_score:.4f}\n")
        f.write(f"# sa: {result.sa_score:.2f}\n")
        f.write(f"# description: {description}\n")


def append_history(entry):
    """Append a JSONL entry for full history."""
    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def format_history_for_ai(recent_results, max_entries=10):
    """Format the recent experiment history for the AI prompt."""
    if not recent_results:
        return "No experiments yet."

    lines = ["Recent experiments (most recent last):"]
    for r in recent_results[-max_entries:]:
        lines.append(
            f"  [{r['status']}] composite={r['composite']:.4f} binding={r['binding']:+.2f} "
            f"qed={r['qed']:.2f} sa={r['sa']:.1f} tox={r['tox']} | {r['smiles']} — {r['description']}"
        )
    return "\n".join(lines)


def ask_ai_for_modification(client, model, current_best, history, target_name):
    """Ask the AI to propose a modification. Returns dict with smiles, description, strategy."""

    user_message = f"""TARGET PROTEIN: {target_name}

CURRENT BEST MOLECULE:
  SMILES: {current_best['smiles']}
  composite_score: {current_best['composite']:.4f}
  binding_affinity: {current_best['binding']:+.2f} kcal/mol
  qed: {current_best['qed']:.3f}
  sa_score: {current_best['sa']:.2f}
  toxicity_alerts: {current_best['tox']}
  lipinski_violations: {current_best['lipinski']}
  description: {current_best['description']}

{format_history_for_ai(history)}

Propose ONE modification to the current best molecule that should improve the composite_score. Output JSON only."""

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text


def run_loop(max_iterations=None, model=DEFAULT_MODEL, verbose=True):
    """Main autonomous loop."""
    # Load config and prepare environment
    config = load_config()
    target_name = config["target"].get("name", config["target"]["pdb_id"])

    print("=" * 70)
    print("MacDock — Autonomous Drug Discovery")
    print("By Emmanuel MacDan")
    print("=" * 70)
    print(f"Target:  {target_name}")
    print(f"Model:   {model}")
    print(f"Backend: {DOCKING_BACKEND}")
    print(f"Max iterations: {max_iterations if max_iterations else 'unlimited'}")
    print()

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    # Prepare protein once
    prepare_environment(config)
    print()

    # Initialize results file
    init_results_file()

    # Initialize Anthropic client
    client = Anthropic()

    # Run baseline
    start_smiles = config["starting_molecule"]["smiles"]
    start_name = config["starting_molecule"]["name"]
    print(f"[baseline] Evaluating {start_name}...")

    try:
        result = evaluate_molecule(start_smiles, config)
        print(
            f"[baseline] composite={result.composite_score:.4f} "
            f"binding={result.binding_affinity:+.2f} qed={result.qed_score:.3f} "
            f"sa={result.sa_score:.2f} tox={result.toxicity_alerts}"
        )
    except Exception as e:
        print(f"[baseline] FAILED: {e}")
        sys.exit(1)

    log_result(0, start_smiles, result, "keep", "baseline", f"baseline {start_name}")

    current_best = {
        "smiles": start_smiles,
        "composite": result.composite_score,
        "binding": result.binding_affinity,
        "qed": result.qed_score,
        "sa": result.sa_score,
        "tox": result.toxicity_alerts,
        "lipinski": result.lipinski_violations,
        "description": f"baseline {start_name}",
    }
    save_best(start_smiles, result, 0, current_best["description"])
    append_history({"iter": 0, "status": "keep", **current_best, "timestamp": datetime.now().isoformat()})

    history = [current_best | {"status": "keep"}]

    # Main loop
    iter_num = 0
    improvements = 0

    while True:
        iter_num += 1
        if max_iterations and iter_num > max_iterations:
            break

        print(f"\n--- Iteration {iter_num} ---")

        # Ask AI for a modification
        try:
            ai_text = ask_ai_for_modification(client, model, current_best, history, target_name)
            proposal = parse_ai_response(ai_text)
            new_smiles = proposal["smiles"]
            description = proposal.get("description", "no description")
            strategy = proposal.get("strategy", "unknown")
        except Exception as e:
            print(f"[iter {iter_num}] AI error: {e}")
            log_result(iter_num, "", None, "ai_error", "none", str(e)[:200])
            time.sleep(2)
            continue

        # Validate SMILES
        mol, canonical = validate_smiles(new_smiles)
        if mol is None:
            print(f"[iter {iter_num}] Invalid SMILES: {new_smiles}")
            log_result(iter_num, new_smiles, None, "invalid_smiles", strategy, description)
            history.append({
                "status": "invalid_smiles", "smiles": new_smiles, "composite": 999,
                "binding": 0, "qed": 0, "sa": 0, "tox": 0, "lipinski": 0,
                "description": description
            })
            continue

        # Dock and score
        try:
            t0 = time.time()
            new_result = evaluate_molecule(canonical, config)
            t1 = time.time()
        except Exception as e:
            print(f"[iter {iter_num}] Docking failed: {e}")
            log_result(iter_num, canonical, None, "dock_error", strategy, description)
            history.append({
                "status": "dock_error", "smiles": canonical, "composite": 999,
                "binding": 0, "qed": 0, "sa": 0, "tox": 0, "lipinski": 0,
                "description": description
            })
            continue

        # Decide: keep or discard?
        improved = new_result.composite_score < current_best["composite"]
        status = "keep" if improved else "discard"

        if verbose:
            arrow = "↓ IMPROVED" if improved else "↑ worse"
            print(
                f"[iter {iter_num}] [{strategy}] composite={new_result.composite_score:.4f} "
                f"binding={new_result.binding_affinity:+.2f} qed={new_result.qed_score:.3f} "
                f"sa={new_result.sa_score:.2f} tox={new_result.toxicity_alerts} "
                f"({t1-t0:.1f}s) {arrow}"
            )
            print(f"           SMILES: {canonical}")
            print(f"           {description}")

        log_result(iter_num, canonical, new_result, status, strategy, description)

        entry = {
            "iter": iter_num,
            "status": status,
            "smiles": canonical,
            "composite": new_result.composite_score,
            "binding": new_result.binding_affinity,
            "qed": new_result.qed_score,
            "sa": new_result.sa_score,
            "tox": new_result.toxicity_alerts,
            "lipinski": new_result.lipinski_violations,
            "description": description,
            "strategy": strategy,
            "dock_time_sec": t1 - t0,
            "timestamp": datetime.now().isoformat(),
        }
        append_history(entry)
        history.append(entry)

        if improved:
            improvements += 1
            current_best = {
                "smiles": canonical,
                "composite": new_result.composite_score,
                "binding": new_result.binding_affinity,
                "qed": new_result.qed_score,
                "sa": new_result.sa_score,
                "tox": new_result.toxicity_alerts,
                "lipinski": new_result.lipinski_violations,
                "description": description,
            }
            save_best(canonical, new_result, iter_num, description)
            print(f"           *** NEW BEST *** (total improvements: {improvements})")

        # Keep history manageable
        if len(history) > 30:
            history = history[-20:]

    # Summary
    print("\n" + "=" * 70)
    print("MacDock Run Complete")
    print("=" * 70)
    print(f"Total iterations:  {iter_num}")
    print(f"Improvements:      {improvements}")
    print(f"Best composite:    {current_best['composite']:.6f}")
    print(f"Best binding:      {current_best['binding']:+.2f} kcal/mol")
    print(f"Best SMILES:       {current_best['smiles']}")
    print(f"Results log:       {RESULTS_PATH}")
    print(f"Best molecule:     {BEST_PATH}")


def main():
    parser = argparse.ArgumentParser(description="MacDock autonomous drug discovery")
    parser.add_argument("--max", type=int, default=None,
                        help="Max iterations (default: unlimited)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Claude model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    args = parser.parse_args()

    try:
        run_loop(
            max_iterations=args.max,
            model=args.model,
            verbose=not args.quiet,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Results saved to results.tsv and best.smi")


if __name__ == "__main__":
    main()
