#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meta_optimizer.py

Bayesian-optimization (Optuna / TPE) driver for meta_absorber.py.
Designed to be run directly from Spyder (F5 / Run File).

Why Optuna + TPE rather than a fixed-dimension GP-BO library (BoTorch,
scikit-optimize, etc.): ribbon_count and notch_count change *how many*
geometry parameters exist (more ribbons -> more ribbon_centers/widths,
more notches per ribbon -> more notch_centers/widths). That's a
conditional / variable-dimension search space, which plain GP kernels
don't model natively. Optuna's "define-by-run" API lets the search space
be built with ordinary Python control flow (if/for) inside the objective
function, and its default TPE sampler handles that natively while still
behaving like a real Bayesian optimizer (kernel-density-based surrogate +
expected-improvement-style acquisition) -- not a blind random/grid search.

Each trial = one call to meta_absorber.FoM(), i.e. one full Lumerical RCWA
sweep over the wavelength range (~1-2 min). Trials run strictly
sequentially (single Lumerical license), matching how
meta_absorber.RCWA_sim() opens and closes its own lumapi.FDTD() session
per call.

HOW TO USE IN SPYDER:
  1. Edit the "=== RUN CONFIGURATION ===" block below to set N_TRIALS,
     STUDY_NAME, OUTPUT_DIR, etc.
  2. Press F5 (or Run > Run File). The study will run and print progress
     to the Spyder console.
  3. Press F5 again later (with the same STUDY_NAME / OUTPUT_DIR) to add
     more trials -- Optuna resumes automatically from the sqlite file.
  4. After a run, the variables `study`, `best_params`, and `df` are
     left in Spyder's namespace for interactive exploration in the console
     or Variable Explorer.
"""

import json
import pickle
import os
import time
import traceback
import pandas as pd 

import numpy as np
import optuna
from optuna.trial import TrialState

import meta_absorber as ma

# =============================================================================
# === RUN CONFIGURATION =======================================================
# Edit these variables before pressing F5.
# =============================================================================

# Number of *new* trials to run this session. If resuming an existing study,
# this many additional trials will be appended on top of what's already done.
N_TRIALS = 1

# Study name -- also used as the stem of the sqlite database filename.
# Change this to start a fresh study; keep it the same to resume a prior run.
STUDY_NAME = "absorber-opt_2nd-run"

# Directory where the sqlite database and best_params JSON are written.
# Use an absolute path to avoid Spyder CWD surprises.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', STUDY_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# RNG seed for the TPE sampler. Set to an integer for reproducible trial
# ordering; leave as None to let Optuna seed from the clock.
SEED = None

# =============================================================================
# === SEARCH-SPACE BOUNDS =====================================================
# All lengths in metres, matching meta_absorber.py's convention.
# Adjust to match your physical / fabrication constraints.
# =============================================================================

MIN_MESA_WIDTH = ma.min_mesa_width       # 50 nm, imported from meta_absorber
MIN_TRENCH_WIDTH    = 50e-9                   # min gap: ribbon/notch edge <-> cell edge or neighbour

PERIOD_X_BOUNDS      = (0.8e-6,  14.0e-6)   # unit cell x period
PERIOD_Y_BOUNDS      = (0.8e-6,  14.0e-6)   # unit cell y period

SL_THICKNESS_BOUNDS  = (0.500e-6, 3.00e-6)  # superlattice layer thickness
AU_THICKNESS_BOUNDS  = (0.050e-6, 0.400e-6)  # Au (superstrate) thickness

RIBBON_COUNT_BOUNDS  = (1, 3)               # inclusive integer range
RIBBON_WIDTH_BOUNDS = (MIN_MESA_WIDTH, 2 * PERIOD_X_BOUNDS[0] / (RIBBON_COUNT_BOUNDS[1] + 1))  # upper is a soft guide; sampling is period-aware so fit-failure
                                                   # pruning is impossible by construction (see _suggest_nonoverlapping_segments)
NOTCH_COUNT_BOUNDS   = (0, 2)               # 0 = unnotched ribbons allowed
NOTCH_WIDTH_BOUNDS = (MIN_TRENCH_WIDTH, 2 * PERIOD_Y_BOUNDS[0] / (NOTCH_COUNT_BOUNDS[1] + 1)) # upper: comfortable fraction of min period_y

# Fixed stack properties -- not searched. Add suggest_categorical calls in
# build_params_from_trial() if you want to search over material choice.
FIXED_LAYER_NAMES     = ['substrate', 'superlattice', 'superstrate']
FIXED_LAYER_MATERIALS = ["GaSb - custom", "InSb30/GaSb70 superlattice", "Au (Gold) - Palik"]
FIXED_LAYER_IS_ETCHED = [False, True, True]
SUBSTRATE_THICKNESS   = 1e-6   # fixed; add bounds above to make it a search param

# Simulation fidelity during optimisation. Using the same values as
# meta_absorber.default_params() gives full fidelity but is slower.
# Reduce Fourier_N / mesh counts for a faster (noisier) optimisation pass,
# then re-run the winning design at full fidelity to verify.
OPT_FOURIER_N       = 50
OPT_XY_MESH         = 90
OPT_Z_MESH          = 55
OPT_WAVELENGTH_RANGE = np.linspace(8e-6, 9.5e-6, 100)


# =============================================================================
# === SEARCH SPACE CONSTRUCTION ===============================================
# (No edits needed below here for a typical run)
# =============================================================================

def _suggest_nonoverlapping_segments(trial, prefix, count, axis_length,
                                      width_bounds, edge_margin):
    """
    Suggests `count` non-overlapping 1D segments (centers + widths) that fit
    within [0, axis_length], ordered left-to-right, each separated from its
    neighbours and from the cell edges by at least `edge_margin`.

    Both widths AND gaps are sampled as normalised fractions of the available
    space (axis_length minus the fixed trench margins), so fit-failure pruning
    is impossible by construction regardless of ribbon/notch count or period.

    The available space is first divided into `2*count + 1` slots: alternating
    gaps (count+1) and widths (count). Raw fractions [0,1] are suggested for
    each, normalised so they sum to 1, then scaled by the available space.
    Widths are then clamped to [width_bounds[0], width_bounds[1]] after
    scaling; if the available space is so small that even the minimum width
    can't fit for all segments, the trial is pruned -- but this is an
    irreducible physical constraint (period genuinely too small for the
    requested number of ribbons/notches), not a sampling artefact.

    Returns (centers: list[float], widths: list[float]).
    Raises optuna.TrialPruned only when the period is physically too small.
    """
    total_margin = edge_margin * (count + 1)
    available    = axis_length - total_margin   # space left after fixed trenches

    # Hard physical constraint: can't fit `count` segments each >= min_width.
    min_width = width_bounds[0]
    if available < count * min_width:
        raise optuna.TrialPruned(
            f"{prefix}: axis_length={axis_length*1e6:.3f}µm leaves only "
            f"{available*1e9:.0f}nm after margins -- not enough for {count} × "
            f"{min_width*1e9:.0f}nm minimum-width segments."
        )

    # Sample raw fractions for widths and gaps jointly, then normalise.
    # Using 2*count+1 slots: [gap_0, width_0, gap_1, width_1, ..., gap_count]
    n_slots = 2 * count + 1
    raw = [trial.suggest_float(f"{prefix}_slot_{i}", 0.0, 1.0) for i in range(n_slots)]
    raw_sum = sum(raw) or 1.0
    fracs = [r / raw_sum for r in raw]

    # Reconstruct widths (odd-indexed slots) and gap fractions (even-indexed).
    raw_widths = [fracs[2*i + 1] * available for i in range(count)]
    raw_gaps   = [fracs[2*i]     * available for i in range(count + 1)]

    # Clamp widths into [width_bounds[0], width_bounds[1]].
    # Clamping shifts some space back into/out of the gaps; redistribute
    # the residual proportionally across gaps to keep the total consistent.
    widths = [np.clip(w, width_bounds[0], width_bounds[1]) for w in raw_widths]
    residual = available - sum(widths) - sum(raw_gaps)
    gap_sum  = sum(raw_gaps) or 1.0
    gaps = [g + residual * (g / gap_sum) for g in raw_gaps]

    centers = []
    cursor  = 0.0
    for i in range(count):
        cursor += edge_margin + gaps[i]
        centers.append(cursor + widths[i] / 2.0)
        cursor += widths[i]

    return centers, widths


def build_params_from_trial(trial):
    """
    Constructs a full meta_absorber-style params dict from an Optuna trial,
    including the conditional ribbon_count / notch_count structure.
    """
    params = {
        'Fourier_N'       : OPT_FOURIER_N,
        'xy_mesh'         : OPT_XY_MESH,
        'z_mesh'          : OPT_Z_MESH,
        'k_mesh'          : 24,
        'wavelength_range': OPT_WAVELENGTH_RANGE,
        'layer_count'     : 3,
        'layer_names'     : list(FIXED_LAYER_NAMES),
        'layer_materials' : list(FIXED_LAYER_MATERIALS),
        'layer_is_etched' : list(FIXED_LAYER_IS_ETCHED),
    }

    period_x = trial.suggest_float("period_x", *PERIOD_X_BOUNDS)
    period_y = trial.suggest_float("period_y", *PERIOD_Y_BOUNDS)
    params['period'] = [period_x, period_y]

    sl_thickness = trial.suggest_float("sl_thickness", *SL_THICKNESS_BOUNDS)
    au_thickness = trial.suggest_float("au_thickness", *AU_THICKNESS_BOUNDS)
    params['layer_thicknesses'] = [SUBSTRATE_THICKNESS, sl_thickness, au_thickness]

    ribbon_count = trial.suggest_int("ribbon_count", *RIBBON_COUNT_BOUNDS)
    params['ribbon_count'] = ribbon_count

    ribbon_centers, ribbon_widths = _suggest_nonoverlapping_segments(
        trial, "ribbon", ribbon_count, period_x, RIBBON_WIDTH_BOUNDS, MIN_TRENCH_WIDTH
    )
    params['ribbon_centers'] = ribbon_centers
    params['ribbon_widths']  = ribbon_widths

    # Note: params carries a single notch_centers/notch_widths set shared
    # across all ribbons (matches meta_absorber.setup() structure). If you
    # want independent notch layouts per ribbon, both setup() and here need
    # extending -- flagged rather than silently assumed.
    notch_count = trial.suggest_int("notch_count", *NOTCH_COUNT_BOUNDS)
    params['notch_count'] = notch_count

    if notch_count > 0:
        # Ensure every ribbon is wide enough to accommodate notches on both
        # sides while leaving MIN_MESA_WIDTH of solid material.
        for r_width in ribbon_widths:
            if r_width <= MIN_MESA_WIDTH:
                raise optuna.TrialPruned(
                    f"ribbon width {r_width*1e9:.0f}nm <= MIN_MESA_WIDTH "
                    f"={MIN_MESA_WIDTH*1e9:.0f}nm; notching would produce "
                    f"zero/negative mesa geometry"
                )
        notch_centers, notch_widths = _suggest_nonoverlapping_segments(
            trial, "notch", notch_count, period_y, NOTCH_WIDTH_BOUNDS, MIN_TRENCH_WIDTH
        )
        params['notch_centers'] = notch_centers
        params['notch_widths']  = notch_widths
    else:
        params['notch_centers'] = []
        params['notch_widths']  = []

    return params


# =============================================================================
# === OBJECTIVE ===============================================================
# =============================================================================

def objective(trial):
    
    t0 = time.time()
    try:
        params = build_params_from_trial(trial)
        A_lambda_pol = ma.RCWA_sim(params)
    except optuna.TrialPruned:
        raise
    except Exception:
        # A Lumerical / geometry failure shouldn't kill the whole study.
        # Log it, prune the trial, and continue.
        print(f"[trial {trial.number}] FAILED after {time.time()-t0:.1f}s:")
        traceback.print_exc()
        raise optuna.TrialPruned("Simulation raised an exception; see console.")

    # Objective: mean absorption averaged over both polarisations and all
    # wavelengths. A_lambda_pol shape = (n_wavelengths, 2).
    mean_absorption = float(np.mean(A_lambda_pol))

    # Extra diagnostics stored on the trial -- visible in study.trials_dataframe()
    # and in Spyder's Variable Explorer after the run.
    trial.set_user_attr("mean_absorption_s", float(np.mean(A_lambda_pol[:, 0])))
    trial.set_user_attr("mean_absorption_p", float(np.mean(A_lambda_pol[:, 1])))
    trial.set_user_attr("min_absorption",    float(np.min(A_lambda_pol)))
    trial.set_user_attr("eval_seconds",      round(time.time() - t0, 1))
    trial.set_user_attr("params_json",       json.dumps(_jsonable(params)))

    print(f"[trial {trial.number:>4d}]  "
          f"mean_absorption = {mean_absorption:.4f}  "
          f"({time.time()-t0:.1f}s)  "
          f"ribbons={params['ribbon_count']}  notches={params['notch_count']}")

    return mean_absorption


def _jsonable(params):
    """Converts a params dict (which may contain numpy arrays/scalars) to a
    json-serialisable form, for storage in Optuna trial user_attrs."""
    out = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (list, tuple)):
            out[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


# =============================================================================
# === MAIN RUN BLOCK ==========================================================
# Executes when you press F5 in Spyder (or run the file any other way).
# `study`, `best_params`, and `df` are left in the namespace afterwards
# for interactive exploration in the Spyder console / Variable Explorer.
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
db_path   = os.path.join(OUTPUT_DIR, "study.db")       
json_path = os.path.join(OUTPUT_DIR, "best_params.json")  

storage   = f"sqlite:///{db_path}"

optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress Optuna's own chatter;
                                                        # our objective() prints per-trial lines instead

sampler = optuna.samplers.TPESampler(
    seed=SEED,
    multivariate=True,  # model correlations between parameters suggested together
    group=True,         # handle conditional/grouped params (ribbon_count, notch_count)
)

study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=storage,
    sampler=sampler,
    direction="maximize",
    load_if_exists=True,   # resumes automatically if the .db already exists
)

n_done = len(study.get_trials(states=[TrialState.COMPLETE]))
print(f"=== Study '{STUDY_NAME}' ===")
print(f"    Loaded {n_done} completed trial(s) from {db_path}")
print(f"    Running {N_TRIALS} new trial(s)...\n")

study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)

# --- Summary -----------------------------------------------------------------
n_complete = len(study.get_trials(states=[TrialState.COMPLETE]))
n_pruned   = len(study.get_trials(states=[TrialState.PRUNED]))

print(f"\n=== Session complete ===")
print(f"    Completed trials : {n_complete}  (pruned/skipped: {n_pruned})")
print(f"    Best mean absorption : {study.best_value:.4f}")
print(f"    Best Optuna params   : {json.dumps(study.best_trial.params, indent=4)}")

# Reconstruct the full meta_absorber params dict from the stored JSON and
# write it out so you can copy-paste it straight into meta_absorber.py.
best_params = json.loads(study.best_trial.user_attrs["params_json"])
best_params["wavelength_range"] = OPT_WAVELENGTH_RANGE  # restore numpy array

with open(json_path, "w") as f:
    json.dump(_jsonable(best_params), f, indent=2)

print(f"\n    Best params dict saved to: {json_path}")
print(f"    Study database          : {db_path}")
print(f"\n    To add more trials: increase N_TRIALS and press F5 again.")
print(f"    To start fresh     : change STUDY_NAME and press F5.")

# Build a summary DataFrame for the Variable Explorer / further analysis.
# Columns: trial number, value, all params, all user_attrs.
df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
df.columns = ["_".join(str(s) for s in col).strip("_") if isinstance(col, tuple) else col
              for col in df.columns]

# Save the full df to disk before any dtype conversion
with open(os.path.join(OUTPUT_DIR, "trials_df.pkl"), "wb") as f:
    pickle.dump(df, f)
# Save best_params (already written as JSON above, nothing extra needed)
# study is already fully persisted in study.db -- no separate save needed

# For the Spyder namespace, convert to a plain dict of lists.
# Spyder can always display dicts, with no dtype compatibility issues.
trials_summary = {col: df[col].tolist() for col in df.columns}

print("\n\t`study`, `best_params`, and `trials_summary` are now in your namespace.")




# =============================================================================
# ###############################################################################
# # To reload into Spyder later, create or paste this into a separate script:
# ###############################################################################
# import json, pickle, os
# import optuna
# import pandas as pd
# import numpy as np
#
# OUTPUT_DIR = r"/path/to/results/absorber-opt_first-run"   # update this
# STUDY_NAME = "absorber-opt_first-run"                     # must match exactly
#
# optuna.logging.set_verbosity(optuna.logging.WARNING)
#
# # Reload study
# storage = f"sqlite:///{os.path.join(OUTPUT_DIR, 'study.db')}"
# study = optuna.load_study(study_name=STUDY_NAME, storage=storage)
#
# # Reload best_params
# with open(os.path.join(OUTPUT_DIR, "best_params.json")) as f:
#     best_params = json.load(f)
# best_params["wavelength_range"] = np.array(best_params["wavelength_range"])
#
# # Reload df (columns already flat, object cols already cast to str)
# with open(os.path.join(OUTPUT_DIR, "trials_df.pkl"), "rb") as f:
#     df = pickle.load(f)
# 
# # For the Spyder namespace, convert to a plain dict of lists.
# # Spyder can always display dicts, with no dtype compatibility issues.
# trials_summary = {col: df[col].tolist() for col in df.columns}
# 
# print("\n\t`study`, `best_params`, and `trials_summary` are now in your namespace.")
# =============================================================================
