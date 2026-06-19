"""
Effective Medium Theory for InSb/GaSb Superlattice
====================================================
Computes effective optical constants (n, k) for a superlattice using
the uniaxial EMT formulas:

  In-plane (parallel):     eps_par  = f*eps_A + (1-f)*eps_B
  Growth direction (perp): eps_perp = 1 / (f/eps_A + (1-f)/eps_B)

where f is the volume fraction (duty cycle) of material A (InSb),
and eps = (n + ik)^2 is the complex permittivity.

The two principal components give the SL uniaxial optical anisotropy.
Also outputs the isotropic average for use as a single effective medium.

Usage:
  python emt_superlattice.py

Inputs (expected columns: wavelength, n, k):
  InSb.csv
  GaSb.csv

Output:
  SL_effective_nk.csv
"""

import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
INSB_FILE  = "InSb.csv"
GASB_FILE  = "GaSb.csv"
OUTPUT_FILE = "SL_effective_nk.txt"

DUTY_CYCLE = 0.30          # fraction of period that is InSb (material A)
WAVELENGTH_UNIT = "um"     # label only — no conversion applied; set to match your CSVs

# ── Load data ─────────────────────────────────────────────────────────────────
insb = pd.read_csv(INSB_FILE)
gasb = pd.read_csv(GASB_FILE)

# Normalise column names to lowercase/strip whitespace
insb.columns = insb.columns.str.strip().str.lower()
gasb.columns = gasb.columns.str.strip().str.lower()

# ── Interpolate GaSb onto InSb wavelength grid (or vice versa) ───────────────
# Use the intersection of the two wavelength ranges to avoid extrapolation.
wl_min = max(insb["wavelength"].min(), gasb["wavelength"].min())
wl_max = min(insb["wavelength"].max(), gasb["wavelength"].max())

insb = insb[(insb["wavelength"] >= wl_min) & (insb["wavelength"] <= wl_max)].copy()
wl   = insb["wavelength"].values          # master wavelength axis

# Interpolate GaSb onto same grid
n_gasb = np.interp(wl, gasb["wavelength"].values, gasb["n"].values)
k_gasb = np.interp(wl, gasb["wavelength"].values, gasb["k"].values)
n_insb = insb["n"].values
k_insb = insb["k"].values

# ── Build complex permittivities ──────────────────────────────────────────────
eps_A = (n_insb + 1j * k_insb) ** 2      # InSb  (material A)
eps_B = (n_gasb + 1j * k_gasb) ** 2      # GaSb  (material B)

f = DUTY_CYCLE

# ── EMT formulas ──────────────────────────────────────────────────────────────
# In-plane (ordinary ray, TE-like)
eps_par  = f * eps_A + (1 - f) * eps_B

# Growth direction (extraordinary ray, TM-like)
eps_perp = 1.0 / (f / eps_A + (1 - f) / eps_B)

# Isotropic average: arithmetic mean of the two principal components
# (useful as a single representative value for normal-incidence simulations)
eps_iso  = (2 * eps_par + eps_perp) / 3.0

# ── Convert back to n, k ──────────────────────────────────────────────────────
def eps_to_nk(eps):
    """Convert complex permittivity to (n, k) with k >= 0."""
    n_complex = np.sqrt(eps)
    # Choose the root with positive imaginary part (physical k)
    mask = np.imag(n_complex) < 0
    n_complex[mask] = -n_complex[mask]
    return np.real(n_complex), np.imag(n_complex)

n_par,  k_par  = eps_to_nk(eps_par.copy())
n_perp, k_perp = eps_to_nk(eps_perp.copy())
n_iso,  k_iso  = eps_to_nk(eps_iso.copy())

# ── Assemble output DataFrame ─────────────────────────────────────────────────
out = pd.DataFrame({
    "wavelength": wl,
    "n_x": n_par,   # in-plane
    "k_x": k_par,
    "n_y": n_par,   # in-plane (same as x by symmetry)
    "k_y": k_par,
    "n_z": n_perp,  # out-of-plane (growth direction)
    "k_z": k_perp,
})

out.to_csv(OUTPUT_FILE, index=False, sep="\t", float_format="%.6f")
print(f"Written {len(out)} rows to {OUTPUT_FILE}")
print(f"\nDuty cycle (InSb fraction): {DUTY_CYCLE:.0%}")
print(f"Wavelength range: {wl.min():.4f} – {wl.max():.4f} {WAVELENGTH_UNIT}")
print(f"\nSample output (first 5 rows):")
print(out.head().to_string(index=False))
