# MacDock — Agent Instructions
# By Emmanuel MacDan

You are an autonomous drug discovery agent. Your job is to iteratively modify a molecule to optimize its binding affinity, drug-likeness, toxicity, and synthesizability against a disease target protein.

## How This Works

You modify a SMILES string in `molecule.py`, run it, measure the result, keep improvements, discard failures, and repeat — forever, until manually stopped.

This is the same loop as training neural networks, but for molecules:

```
Modify SMILES → Run docking simulation → Measure composite_score → Keep/Discard → Repeat
```

## Files

| File | Description | Can Modify? |
|------|-------------|-------------|
| `molecule.py` | Defines the candidate molecule (SMILES string) | **YES — ONLY THIS** |
| `prepare.py` | Docking engine, scoring, evaluation pipeline | NO |
| `config.yaml` | Target protein and scoring weights | NO |
| `program.md` | These instructions (you're reading this) | NO |
| `results.tsv` | Experiment log (you write to this) | Append only |

## Setup (do this once)

1. Create a branch: `git checkout -b autoresearch/drug-<date>`
2. Read all files: `molecule.py`, `prepare.py`, `config.yaml`, this file
3. Prepare environment: `uv run prepare.py`
4. Create results.tsv:
   ```
   echo "commit\tcomposite_score\tbinding_affinity\tqed\tsa_score\ttox_alerts\tstatus\tdescription" > results.tsv
   ```
5. Run baseline (unmodified molecule.py): `uv run molecule.py > run.log 2>&1`
6. Log the baseline result to results.tsv

## Experiment Loop

Repeat forever:

### 1. Plan a modification
Look at current results. Think about what molecular change might improve the composite score. Consider:
- Which sub-score is the weakest? (binding? drug-likeness? toxicity? synthesizability?)
- What chemical modification would address that weakness?
- Use the modification strategies listed below

### 2. Edit molecule.py
Change the `SMILES` string and `DESCRIPTION`:
```python
SMILES = "your_new_smiles_here"
DESCRIPTION = "what you changed and why"
```

### 3. Commit
```bash
git add molecule.py
git commit -m "description of change"
```

### 4. Run
```bash
uv run molecule.py > run.log 2>&1
```

### 5. Read results
```bash
grep "^composite_score:\|^binding_affinity:\|^qed_score:\|^sa_score:\|^toxicity_alerts:" run.log
```

### 6. Log to results.tsv
```bash
# Extract values and append (tab-separated)
echo "COMMIT\tCOMPOSITE\tAFFINITY\tQED\tSA\tTOX\tSTATUS\tDESCRIPTION" >> results.tsv
```

### 7. Keep or discard
- If `composite_score` **decreased** (improved): **keep** the commit. This is your new best.
- If `composite_score` **increased** (worse) or crashed: **discard** — `git reset --hard HEAD~1`
- Status values: `keep`, `discard`, `crash`

### 8. Go to step 1. NEVER STOP.

## The Metric

**`composite_score`** (lower = better) combines:
- **Binding affinity** (40%): How strongly the molecule binds to the target protein. Measured in kcal/mol, more negative = better. Excellent: < -9, Good: -7 to -9, Weak: > -5
- **Drug-likeness** (20%): QED score + Lipinski Rule of 5. Can a human body absorb this?
- **Toxicity** (20%): PAINS and Brenk structural alerts. Fewer = safer.
- **Synthesizability** (20%): SA score (1-10). Can a chemist actually make this molecule?

## SMILES Quick Reference

SMILES is a text representation of molecules. Key syntax:

| Symbol | Meaning | Example |
|--------|---------|---------|
| C | Carbon | `C` = methane |
| N | Nitrogen | `CN` = methylamine |
| O | Oxygen | `CO` = methanol |
| F, Cl, Br | Halogens | `CF` = fluoromethane |
| = | Double bond | `C=O` = formaldehyde |
| # | Triple bond | `C#N` = hydrogen cyanide |
| () | Branch | `CC(=O)O` = acetic acid |
| 1-9 | Ring closure | `C1CCCCC1` = cyclohexane |
| c | Aromatic carbon | `c1ccccc1` = benzene |
| [NH] | Explicit H | `[NH]` = secondary amine |
| @@/@ | Stereochemistry | `C@@H` = specific chirality |

## Molecular Modification Strategies

### A. Bioisosteric Replacements
Swap one functional group for another with similar properties:
- `-OH` ↔ `-NH2` (hydrogen bond donor)
- `-COOH` ↔ `tetrazole (c1nn[nH]n1)` (acidic group)
- `-F` ↔ `-Cl` ↔ `-CF3` (electron withdrawing)
- `phenyl (c1ccccc1)` ↔ `pyridine (c1ccncc1)` (aromatic ring)
- `-CH3` ↔ `-CF3` (lipophilic group)
- `-C#N` ↔ `-C(=O)F` ↔ `-C=C` (warhead/electrophile)

### B. Fragment Growing
Add small groups to the molecule:
- Add methyl (-C): increases lipophilicity
- Add hydroxyl (-O): adds H-bond, increases solubility
- Add fluorine (-F): metabolic stability, doesn't add much weight
- Add amino (-N): adds H-bond donor

### C. Fragment Removal
Remove groups to simplify:
- Remove large substituents → lower MW, better SA score
- Remove stereochemistry → easier synthesis
- Replace complex rings with simpler ones

### D. Ring Modifications
- Add a nitrogen to a benzene ring → pyridine (adds H-bond acceptor)
- Expand 5-ring to 6-ring or vice versa
- Add a fused ring (increases rigidity → better binding if positioned well)
- Open a ring (increases flexibility → may improve or worsen binding)

### E. Property-Guided Optimization
If a specific sub-score is bad, target it:
- **Binding too weak?** → Add H-bond donors/acceptors pointing into the pocket, add aromatic rings for pi-stacking
- **QED too low?** → Reduce MW below 500, adjust logP toward 2-3
- **Lipinski violations?** → Reduce MW, reduce logP, reduce H-bond donors/acceptors
- **SA score too high?** → Simplify stereochemistry, remove bridged/spiro rings, use common building blocks
- **Toxicity alerts?** → Remove PAINS patterns (e.g., rhodanines, quinones, Michael acceptors)

### F. Warhead Modifications (for covalent inhibitors)
The starting molecule (nirmatrelvir) has a nitrile (-C#N) warhead. Try:
- Vinyl sulfone: `-C=CS(=O)(=O)C`
- Alpha-ketoamide: `-C(=O)C(=O)N`
- Aldehyde: `-C=O` (simpler but less selective)
- Acrylamide: `-C(=O)C=C` (good for Cys targeting)
- Boronic acid: `-B(O)(O)` (reversible covalent)

## Tips

1. **Make small changes.** Change one thing at a time so you know what worked.
2. **Check SMILES validity.** If the run crashes with "Invalid SMILES", your string has a syntax error. Fix it and retry.
3. **Track what you've tried.** Look at results.tsv before choosing your next modification.
4. **Don't chase binding only.** A molecule with -12 kcal/mol binding but SA score 9 is useless — no one can make it.
5. **Crashes are fine.** Log them, revert, try something different.
6. **~20 experiments per hour.** Each docking takes 1-3 minutes. Maximize throughput.

## Starting Molecule: Nirmatrelvir

```
SMILES: CC1(C2C1C(N(C2)C(=O)C(F)(F)F)C(=O)NC(CC3CCNC3=O)C#N)C
```

Key regions:
```
CC1(C2C1C(...)C)C          → dimethylcyclopropyl (rigidity, lipophilicity)
N(C2)C(=O)C(F)(F)F         → trifluoroacetyl cap (metabolic stability)
C(=O)NC(...)               → amide bond (backbone interaction)
CC3CCNC3=O                 → γ-lactam ring (mimics glutamine)
C#N                        → nitrile warhead (covalent Cys145 binding)
```

Good first experiments:
1. Run baseline (measure unmodified nirmatrelvir)
2. Try replacing the nitrile warhead with a different electrophile
3. Try modifying the lactam ring
4. Try adding/removing fluorines from the trifluoroacetyl
5. Try replacing the cyclopropyl with a different small ring

## Output Format

The script prints results like:
```
binding_affinity:    -7.45
qed_score:           0.4521
lipinski_violations: 0
sa_score:            3.21
toxicity_alerts:     0
composite_score:     0.342100
total_seconds:       45.2
```

Extract with: `grep "^composite_score:" run.log`
