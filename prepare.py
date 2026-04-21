"""
MacDock — Preparation & Evaluation Infrastructure
==================================================
By Emmanuel MacDan

DO NOT MODIFY THIS FILE. The agent only modifies molecule.py.

This file provides:
  - Protein download and receptor preparation
  - SMILES validation and 3D conformer generation
  - Molecular docking (AutoDock Vina or RDKit fallback)
  - Scoring: binding affinity, drug-likeness, toxicity, synthesizability
  - Top-level evaluate_molecule() function

Usage:
  uv run prepare.py              # Download protein & prepare receptor
  uv run prepare.py --validate   # Validate environment is ready
"""

import os
import sys
import time
import math
import yaml
import shutil
import hashlib
import tempfile
import subprocess
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import requests

# ---------------------------------------------------------------------------
# RDKit imports
# ---------------------------------------------------------------------------
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Docking backend detection
# ---------------------------------------------------------------------------
DOCKING_BACKEND = "none"

try:
    from vina import Vina
    DOCKING_BACKEND = "vina_python"
except ImportError:
    # Try command-line vina
    if shutil.which("vina"):
        DOCKING_BACKEND = "vina_cli"
    elif shutil.which("smina"):
        DOCKING_BACKEND = "smina_cli"
    else:
        DOCKING_BACKEND = "rdkit_only"

# Try meeko for ligand preparation
MEEKO_AVAILABLE = False
try:
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    MEEKO_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
TIME_BUDGET = 180  # 3 minutes max for docking

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class MoleculeResult:
    smiles: str
    canonical_smiles: str
    binding_affinity: float    # kcal/mol (more negative = better)
    qed_score: float           # 0-1 (higher = more drug-like)
    lipinski_violations: int   # 0-5
    sa_score: float            # 1-10 (lower = easier to synthesize)
    toxicity_alerts: int       # count of PAINS/Brenk hits
    molecular_weight: float
    logp: float
    num_atoms: int
    composite_score: float     # weighted combination (lower = better)

# ---------------------------------------------------------------------------
# SA Score (Ertl & Schuffenhauer, 2009)
# Simplified implementation using RDKit fragment contributions
# ---------------------------------------------------------------------------
def calculate_sa_score(mol):
    """Synthetic Accessibility score: 1 (easy) to 10 (hard)."""
    try:
        # Use fragment-based heuristic
        num_atoms = mol.GetNumHeavyAtoms()
        ring_info = mol.GetRingInfo()
        num_rings = ring_info.NumRings()
        num_stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
        num_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        num_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        num_macrocycles = sum(1 for r in ring_info.AtomRings() if len(r) > 8)

        # Complexity penalties
        score = 1.0
        score += num_stereo * 0.5
        score += num_spiro * 0.8
        score += num_bridgehead * 0.8
        score += num_macrocycles * 1.0

        # Size penalty
        if num_atoms > 35:
            score += (num_atoms - 35) * 0.1

        # Ring complexity
        if num_rings > 4:
            score += (num_rings - 4) * 0.5

        # Clip to 1-10 range
        return max(1.0, min(10.0, score))
    except Exception:
        return 5.0  # default mid-range

# ---------------------------------------------------------------------------
# Toxicity filters (PAINS + structural alerts)
# ---------------------------------------------------------------------------
_FILTER_CATALOG = None

def get_filter_catalog():
    global _FILTER_CATALOG
    if _FILTER_CATALOG is None:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        _FILTER_CATALOG = FilterCatalog(params)
    return _FILTER_CATALOG

def count_toxicity_alerts(mol):
    """Count PAINS and Brenk structural alert matches."""
    catalog = get_filter_catalog()
    entries = catalog.GetMatches(mol)
    return len(entries)

# ---------------------------------------------------------------------------
# Lipinski's Rule of Five
# ---------------------------------------------------------------------------
def count_lipinski_violations(mol):
    """Count violations of Lipinski's Rule of Five."""
    violations = 0
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)

    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1

    return violations

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(config_path=None):
    """Load config.yaml from project directory."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Protein download and preparation
# ---------------------------------------------------------------------------
def get_cache_dir(config):
    """Return resolved cache directory path."""
    cache = os.path.expanduser(config.get("cache_dir", "~/.cache/macdock"))
    os.makedirs(cache, exist_ok=True)
    return cache

def download_protein(pdb_id, cache_dir):
    """Download PDB structure from RCSB."""
    protein_dir = os.path.join(cache_dir, "proteins")
    os.makedirs(protein_dir, exist_ok=True)

    pdb_path = os.path.join(protein_dir, f"{pdb_id}.pdb")

    if os.path.exists(pdb_path):
        print(f"  Protein {pdb_id} already cached at {pdb_path}")
        return pdb_path

    url = PDB_DOWNLOAD_URL.format(pdb_id=pdb_id)
    print(f"  Downloading {pdb_id} from RCSB...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    with open(pdb_path, "w") as f:
        f.write(resp.text)

    print(f"  Saved to {pdb_path}")
    return pdb_path

def clean_protein(pdb_path, chain="A"):
    """Remove water, heteroatoms (keep protein), select chain. Returns cleaned PDB text."""
    lines = []
    with open(pdb_path) as f:
        for line in f:
            record = line[:6].strip()
            if record in ("ATOM",):
                # Filter by chain if specified
                line_chain = line[21].strip()
                if chain and line_chain != chain:
                    continue
                lines.append(line)
            elif record in ("TER", "END"):
                lines.append(line)

    return "".join(lines)

def prepare_receptor_pdbqt(pdb_path, chain, cache_dir):
    """Prepare receptor PDBQT for docking. Returns path to PDBQT file."""
    pdbqt_path = os.path.join(cache_dir, "proteins",
                               f"{os.path.basename(pdb_path).replace('.pdb', '')}_receptor.pdbqt")

    if os.path.exists(pdbqt_path):
        print(f"  Receptor PDBQT already prepared at {pdbqt_path}")
        return pdbqt_path

    # Clean protein first
    cleaned = clean_protein(pdb_path, chain)

    # Try openbabel for PDB -> PDBQT conversion
    if shutil.which("obabel"):
        cleaned_pdb = pdbqt_path.replace(".pdbqt", "_clean.pdb")
        with open(cleaned_pdb, "w") as f:
            f.write(cleaned)

        result = subprocess.run(
            ["obabel", cleaned_pdb, "-O", pdbqt_path, "-xr", "-p", "7.4"],
            capture_output=True, text=True
        )
        if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0:
            print(f"  Receptor PDBQT prepared via OpenBabel: {pdbqt_path}")
            return pdbqt_path

    # Fallback: manual PDBQT conversion (simplified)
    print("  Converting to PDBQT (simplified converter)...")
    pdbqt_lines = []
    for line in cleaned.splitlines():
        if line.startswith("ATOM"):
            # Add AutoDock atom type (simplified: use element symbol)
            element = line[76:78].strip()
            if not element:
                element = line[12:16].strip()[0]
            # Standard PDBQT: add charge (0.0) and atom type
            pdbqt_line = line[:54] + "  0.00  0.00"
            pdbqt_line = pdbqt_line.ljust(77) + f" {element:<2s}"
            pdbqt_lines.append(pdbqt_line)
        elif line.startswith("TER") or line.startswith("END"):
            pdbqt_lines.append(line)

    with open(pdbqt_path, "w") as f:
        f.write("\n".join(pdbqt_lines) + "\n")

    print(f"  Receptor PDBQT prepared: {pdbqt_path}")
    return pdbqt_path

# ---------------------------------------------------------------------------
# Ligand preparation
# ---------------------------------------------------------------------------
def smiles_to_3d(smiles):
    """Convert SMILES to 3D RDKit Mol with optimized geometry."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status == -1:
        # Try with more permissive settings
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol, params)
        if status == -1:
            return None

    # Optimize geometry
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass  # Use unoptimized coords

    return mol

def mol_to_pdbqt_string(mol):
    """Convert RDKit Mol (with 3D coords) to PDBQT string."""
    if MEEKO_AVAILABLE:
        try:
            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol)
            pdbqt_string = PDBQTWriterLegacy.write_string(mol_setups[0])[0]
            return pdbqt_string
        except Exception:
            pass

    # Fallback: write PDB then convert
    pdb_block = Chem.MolToPDBBlock(mol)
    if pdb_block is None:
        return None

    # Try openbabel
    if shutil.which("obabel"):
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
            f.write(pdb_block)
            pdb_tmp = f.name

        pdbqt_tmp = pdb_tmp.replace(".pdb", ".pdbqt")
        subprocess.run(["obabel", pdb_tmp, "-O", pdbqt_tmp, "-p", "7.4"],
                       capture_output=True, text=True)
        os.unlink(pdb_tmp)

        if os.path.exists(pdbqt_tmp):
            with open(pdbqt_tmp) as f:
                pdbqt_str = f.read()
            os.unlink(pdbqt_tmp)
            return pdbqt_str

    # Minimal PDBQT from PDB
    lines = []
    for line in pdb_block.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            element = line[76:78].strip()
            if not element:
                element = line[12:16].strip()[0]
            pdbqt_line = line[:54] + "  0.00  0.00"
            pdbqt_line = pdbqt_line.ljust(77) + f" {element:<2s}"
            lines.append(pdbqt_line)
    lines.append("TORSDOF 0")
    return "\n".join(lines) + "\n"

# ---------------------------------------------------------------------------
# Docking
# ---------------------------------------------------------------------------
def dock_vina_python(receptor_pdbqt, ligand_pdbqt_str, config):
    """Dock using Vina Python bindings. Returns best affinity (kcal/mol)."""
    site = config["target"]["binding_site"]

    v = Vina(sf_name="vina")
    v.set_receptor(receptor_pdbqt)

    # Write ligand to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", mode="w", delete=False) as f:
        f.write(ligand_pdbqt_str)
        lig_path = f.name

    try:
        v.set_ligand_from_file(lig_path)
        v.compute_vina_maps(
            center=[site["center_x"], site["center_y"], site["center_z"]],
            box_size=[site["size_x"], site["size_y"], site["size_z"]]
        )

        dock_cfg = config.get("docking", {})
        v.dock(
            exhaustiveness=dock_cfg.get("exhaustiveness", 8),
            n_poses=dock_cfg.get("num_modes", 5),
        )

        energies = v.energies()
        if energies is not None and len(energies) > 0:
            return float(energies[0][0])  # Best pose affinity
    finally:
        os.unlink(lig_path)

    return 0.0

def dock_vina_cli(receptor_pdbqt, ligand_pdbqt_str, config, cmd="vina"):
    """Dock using Vina/smina CLI. Returns best affinity (kcal/mol)."""
    site = config["target"]["binding_site"]
    dock_cfg = config.get("docking", {})

    with tempfile.TemporaryDirectory() as tmpdir:
        lig_path = os.path.join(tmpdir, "ligand.pdbqt")
        out_path = os.path.join(tmpdir, "out.pdbqt")

        with open(lig_path, "w") as f:
            f.write(ligand_pdbqt_str)

        args = [
            cmd,
            "--receptor", receptor_pdbqt,
            "--ligand", lig_path,
            "--out", out_path,
            "--center_x", str(site["center_x"]),
            "--center_y", str(site["center_y"]),
            "--center_z", str(site["center_z"]),
            "--size_x", str(site["size_x"]),
            "--size_y", str(site["size_y"]),
            "--size_z", str(site["size_z"]),
            "--exhaustiveness", str(dock_cfg.get("exhaustiveness", 8)),
            "--num_modes", str(dock_cfg.get("num_modes", 5)),
            "--seed", str(dock_cfg.get("seed", 42)),
        ]

        result = subprocess.run(
            args, capture_output=True, text=True,
            timeout=dock_cfg.get("timeout_seconds", TIME_BUDGET)
        )

        # Parse output for affinity
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[0] == "1":
                try:
                    return float(parts[1])
                except ValueError:
                    continue

    return 0.0

def dock_rdkit_fallback(mol, pdb_path, config):
    """Fallback scoring using RDKit descriptors (no actual docking).
    Returns estimated binding affinity based on molecular properties."""

    # Property-based scoring heuristic
    # This is NOT real docking — just a rough estimate based on drug-likeness
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    tpsa = Descriptors.TPSA(mol)
    rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumAromaticRings(mol)

    # Heuristic: estimate binding energy from properties
    # More HBA/HBD = more hydrogen bonds = better binding
    # Moderate logP = good membrane/binding balance
    # Aromatic rings = pi-stacking interactions
    score = -3.0  # baseline
    score -= min(hba, 8) * 0.3   # H-bond acceptors
    score -= min(hbd, 4) * 0.4   # H-bond donors
    score -= min(rings, 3) * 0.5 # aromatic stacking
    score += abs(logp - 2.5) * 0.2  # penalty for extreme logP
    score += max(0, rotatable - 8) * 0.15  # flexibility penalty

    # Clip to realistic range
    return max(-12.0, min(-1.0, score))

def run_docking(receptor_pdbqt, mol_3d, smiles, config, pdb_path=None):
    """Run molecular docking. Returns binding affinity in kcal/mol."""

    if DOCKING_BACKEND == "vina_python":
        pdbqt_str = mol_to_pdbqt_string(mol_3d)
        if pdbqt_str:
            return dock_vina_python(receptor_pdbqt, pdbqt_str, config)

    if DOCKING_BACKEND == "vina_cli":
        pdbqt_str = mol_to_pdbqt_string(mol_3d)
        if pdbqt_str:
            return dock_vina_cli(receptor_pdbqt, pdbqt_str, config, cmd="vina")

    if DOCKING_BACKEND == "smina_cli":
        pdbqt_str = mol_to_pdbqt_string(mol_3d)
        if pdbqt_str:
            return dock_vina_cli(receptor_pdbqt, pdbqt_str, config, cmd="smina")

    # Fallback: RDKit property-based estimation
    mol_no_h = Chem.RemoveHs(mol_3d)
    return dock_rdkit_fallback(mol_no_h, pdb_path, config)

# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------
def compute_composite_score(binding_affinity, qed_score, lipinski_violations,
                            sa_score, toxicity_alerts, weights):
    """Compute weighted composite score. Lower = better."""

    # Normalize binding affinity: -12 kcal/mol (excellent) → 0, -2 (weak) → 1
    norm_affinity = max(0.0, min(1.0, (binding_affinity + 12.0) / 10.0))

    # QED: 0-1, higher is more drug-like → invert so lower = better
    norm_qed = 1.0 - qed_score

    # Lipinski: 0 violations = 0 (best), 4+ = 1 (worst)
    norm_lipinski = min(1.0, lipinski_violations / 4.0)

    # Drug-likeness = average of QED and Lipinski
    norm_druglike = 0.5 * norm_qed + 0.5 * norm_lipinski

    # SA Score: 1 (easy) to 10 (hard) → normalize to 0-1
    norm_sa = (sa_score - 1.0) / 9.0

    # Toxicity alerts: 0 = 0, 5+ = 1
    norm_tox = min(1.0, toxicity_alerts / 5.0)

    composite = (
        weights.get("binding_affinity", 0.4) * norm_affinity +
        weights.get("drug_likeness", 0.2) * norm_druglike +
        weights.get("synthesizability", 0.2) * norm_sa +
        weights.get("toxicity", 0.2) * norm_tox
    )

    return composite

# ---------------------------------------------------------------------------
# SMILES validation
# ---------------------------------------------------------------------------
def validate_smiles(smiles):
    """Validate SMILES and return (mol, canonical_smiles) or (None, None)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    canonical = Chem.MolToSmiles(mol)
    return mol, canonical

# ---------------------------------------------------------------------------
# Top-level: prepare environment
# ---------------------------------------------------------------------------
def prepare_environment(config):
    """Download protein and prepare receptor. Called once per target."""
    cache_dir = get_cache_dir(config)
    pdb_id = config["target"]["pdb_id"]
    chain = config["target"].get("chain", "A")

    print(f"Target: {config['target'].get('name', pdb_id)}")
    print(f"PDB: {pdb_id}, Chain: {chain}")
    print(f"Docking backend: {DOCKING_BACKEND}")
    print(f"Meeko available: {MEEKO_AVAILABLE}")
    print()

    # Download protein
    pdb_path = download_protein(pdb_id, cache_dir)

    # Prepare receptor
    receptor_pdbqt = prepare_receptor_pdbqt(pdb_path, chain, cache_dir)

    return pdb_path, receptor_pdbqt

# ---------------------------------------------------------------------------
# Top-level: evaluate molecule
# ---------------------------------------------------------------------------
def evaluate_molecule(smiles, config):
    """Full evaluation pipeline. Returns MoleculeResult."""
    cache_dir = get_cache_dir(config)
    pdb_id = config["target"]["pdb_id"]
    chain = config["target"].get("chain", "A")
    weights = config.get("scoring", {}).get("weights", {})

    # Paths
    pdb_path = os.path.join(cache_dir, "proteins", f"{pdb_id}.pdb")
    receptor_pdbqt = os.path.join(cache_dir, "proteins", f"{pdb_id}_receptor.pdbqt")

    # Validate SMILES
    mol, canonical = validate_smiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Properties
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    num_atoms = mol.GetNumHeavyAtoms()
    qed_score = QED.qed(mol)
    lipinski = count_lipinski_violations(mol)
    sa = calculate_sa_score(mol)
    tox = count_toxicity_alerts(mol)

    # 3D conformer for docking
    mol_3d = smiles_to_3d(canonical)
    if mol_3d is None:
        raise ValueError(f"Could not generate 3D conformer for: {canonical}")

    # Docking
    binding_affinity = run_docking(receptor_pdbqt, mol_3d, canonical, config, pdb_path)

    # Composite score
    composite = compute_composite_score(
        binding_affinity, qed_score, lipinski, sa, tox, weights
    )

    return MoleculeResult(
        smiles=smiles,
        canonical_smiles=canonical,
        binding_affinity=binding_affinity,
        qed_score=qed_score,
        lipinski_violations=lipinski,
        sa_score=sa,
        toxicity_alerts=tox,
        molecular_weight=mw,
        logp=logp,
        num_atoms=num_atoms,
        composite_score=composite,
    )

# ---------------------------------------------------------------------------
# CLI: prepare.py as standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config = load_config()

    if "--validate" in sys.argv:
        print("Validating environment...")
        print(f"  Docking backend: {DOCKING_BACKEND}")
        print(f"  Meeko: {MEEKO_AVAILABLE}")

        # Test SMILES
        start_smiles = config["starting_molecule"]["smiles"]
        mol, canonical = validate_smiles(start_smiles)
        if mol:
            print(f"  Starting molecule: {config['starting_molecule']['name']}")
            print(f"  SMILES valid: {canonical}")
            print(f"  Atoms: {mol.GetNumHeavyAtoms()}, MW: {Descriptors.MolWt(mol):.1f}")
        else:
            print("  ERROR: Invalid starting SMILES!")
            sys.exit(1)

        # Check protein
        cache_dir = get_cache_dir(config)
        pdb_id = config["target"]["pdb_id"]
        pdb_path = os.path.join(cache_dir, "proteins", f"{pdb_id}.pdb")
        pdbqt_path = os.path.join(cache_dir, "proteins", f"{pdb_id}_receptor.pdbqt")
        print(f"  Protein cached: {os.path.exists(pdb_path)}")
        print(f"  Receptor PDBQT: {os.path.exists(pdbqt_path)}")

        print("\nEnvironment OK!")
    else:
        print("=" * 60)
        print("MacDock — Environment Setup")
        print("=" * 60)
        print()

        pdb_path, receptor_pdbqt = prepare_environment(config)

        print()
        print("Setup complete!")
        print(f"  Protein: {pdb_path}")
        print(f"  Receptor: {receptor_pdbqt}")
        print()

        # Quick test with starting molecule
        start_smiles = config["starting_molecule"]["smiles"]
        mol, canonical = validate_smiles(start_smiles)
        print(f"Starting molecule: {config['starting_molecule']['name']}")
        print(f"  SMILES: {canonical}")
        print(f"  Atoms: {mol.GetNumHeavyAtoms()}, MW: {Descriptors.MolWt(mol):.1f}")
        print()
        print("Ready to run: uv run molecule.py")
