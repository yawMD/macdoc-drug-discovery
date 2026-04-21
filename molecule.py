"""
MacDock — Molecule Definition
==============================
By Emmanuel MacDan

This is the ONLY file the agent modifies.

The agent changes the SMILES string below to propose new drug candidates.
The evaluation pipeline (prepare.py) handles all scoring automatically.

Usage: uv run molecule.py
"""

from prepare import (
    load_config, prepare_environment, evaluate_molecule,
    validate_smiles, DOCKING_BACKEND
)
from rdkit.Chem import Descriptors

# =========================================================================
# MOLECULE DEFINITION — MODIFY THIS SECTION
# =========================================================================

# The candidate molecule as a SMILES string.
# Starting point: Nirmatrelvir (Paxlovid active ingredient)
# - Known SARS-CoV-2 Mpro inhibitor
# - Covalent inhibitor with nitrile warhead (-C#N)
# - Key modifiable regions:
#   * Nitrile warhead (C#N)
#   * Trifluoroacetyl cap (C(=O)C(F)(F)F)
#   * Dimethylcyclopropyl group (CC1(C2C1...)C)
#   * Lactam ring (CC3CCNC3=O)

SMILES = "CC1(C2C1C(N(C2)C(=O)C(F)(F)F)C(=O)NC(Cc3cc(C)ncc3)C#N)C"

# Brief description of what was changed from the previous iteration
DESCRIPTION = "exp7 best molecule - methylpyridine (OpenBabel receptor)"

# =========================================================================
# EVALUATION HARNESS — DO NOT MODIFY BELOW THIS LINE
# =========================================================================

if __name__ == "__main__":
    import sys
    import time

    config = load_config()

    print("=" * 60)
    print("MacDock — Molecule Evaluation")
    print("=" * 60)
    print()
    print(f"Description: {DESCRIPTION}")
    print(f"SMILES:      {SMILES}")
    print(f"Backend:     {DOCKING_BACKEND}")
    print()

    # Validate SMILES
    mol, canonical = validate_smiles(SMILES)
    if mol is None:
        print("ERROR: Invalid SMILES string")
        sys.exit(1)

    mw = Descriptors.MolWt(mol)
    print(f"Canonical:   {canonical}")
    print(f"Atoms: {mol.GetNumHeavyAtoms()}, MW: {mw:.1f}")
    print()

    # Prepare environment (downloads protein if needed)
    print("Preparing environment...")
    prepare_environment(config)
    print()

    # Run evaluation
    print("Evaluating molecule...")
    t0 = time.time()

    try:
        result = evaluate_molecule(SMILES, config)
        t1 = time.time()

        print()
        print("--- RESULTS ---")
        print(f"binding_affinity:    {result.binding_affinity:.2f}")
        print(f"qed_score:           {result.qed_score:.4f}")
        print(f"lipinski_violations: {result.lipinski_violations}")
        print(f"sa_score:            {result.sa_score:.2f}")
        print(f"toxicity_alerts:     {result.toxicity_alerts}")
        print(f"molecular_weight:    {result.molecular_weight:.1f}")
        print(f"logp:                {result.logp:.2f}")
        print(f"composite_score:     {result.composite_score:.6f}")
        print(f"total_seconds:       {t1 - t0:.1f}")

    except Exception as e:
        t1 = time.time()
        print()
        print(f"ERROR: {e}")
        print(f"composite_score: 999.000000")
        print(f"binding_affinity: 0.00")
        print(f"total_seconds: {t1 - t0:.1f}")
        sys.exit(1)
