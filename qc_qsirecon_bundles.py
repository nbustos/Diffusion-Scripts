# qc_qsirecon_bundles.py - QC of QSIRecon DSI Studio bundles (MRtrix .tck.gz) for the COMBO cohorts. Run with --mode mom or --mode baby.
# Moms: bundles are space-T1w, overlaid on the raw BIDS T1w (QSIPrep only wrote ACPC anatomy; no T1w-space mask, so only an outside-image check). Babies: bundles are space-ACPC, overlaid on the ACPC QSIPrep T1w (T2w fallback).
# Per subject/session: streamline counts (read from .tck headers), fraction of streamline points outside the brain mask, and a PNG of bundle density on the anatomy.
# Flags empty bundles, bundles missing relative to the rest of the cohort, and bundles outside the brain mask. Writes CSVs + PNGs to a separate QC folder (inputs are read-only).
# Usage: python3 qc_qsirecon_bundles.py --mode mom [--subjects 1755138P R1762041 ...] [--out DIR] [--limit N]; needs numpy, nibabel, matplotlib (no MRtrix needed).


import argparse
import csv
import gzip
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.streamlines import TckFile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from combo_qsirecon import BASE_BIDS_DIR, INPUT_QSIPREP_MOMS_DIR, INPUT_QSIPREP_BABIES_DIR, OUTPUT_DERIVATIVES_DIR, CHECK_OUTPUT_DIR

MODES = {
    # space: the "space-" entity in the bundle filenames; anat_kinds: QSIPrep anatomical images to try, in order
    "mom":  dict(space="T1w",  qsiprep=INPUT_QSIPREP_MOMS_DIR,   anat_kinds=None,  # None -> raw BIDS T1w (QSIPrep anat is ACPC only)
                 is_cohort=lambda i: i.lower().endswith("p")),
    "baby": dict(space="ACPC", qsiprep=INPUT_QSIPREP_BABIES_DIR, anat_kinds=["desc-preproc_T1w", "desc-preproc_T2w"],
                 is_cohort=lambda i: i[-1].isdigit()),
}
RAW_BIDS_DIR = BASE_BIDS_DIR
QSIRECON_DIR = OUTPUT_DERIVATIVES_DIR  # overridable with --qsirecon-dir
DEFAULT_OUT = Path("/Marshlab/data/COMBO/diffusion_qc")
KEY_BUNDLES = ["CorpusCallosum", "ArcuateFasciculus", "CorticospinalTract", "OpticRadiation", "InferiorFrontoOccipital", "Cingulum", "UncinateFasciculus"]
EXTRA_BUNDLE_DIRS = ["UF_L", "UF_R"]  # extra derivatives/ folders holding flat <sub>_<ses>_..._bundle-*.tck.gz files (uncinate)
MAX_OUTSIDE_MASK = 0.10  # flag a bundle if more than this fraction of its points fall outside the brain mask (a starting guess)

def tck_count(path):
    """Streamline count from the .tck(.gz) text header (no need to read the streamlines)."""
    with gzip.open(path, "rb") as f:
        for line in f:
            line = line.decode(errors="ignore").strip()
            if line.lower().startswith("count:"): return int(line.split(":")[1])
            if line == "END": break
    return None

def load_points(path):
    """All streamline points (N x 3, world mm RAS) of a .tck.gz file."""
    with tempfile.TemporaryDirectory() as tmp:
        tck = Path(tmp) / "bundle.tck"
        with gzip.open(path, "rb") as src, open(tck, "wb") as dst: shutil.copyfileobj(src, dst)
        return TckFile.load(str(tck), lazy_load=False).streamlines.get_data()

def find_anat(qsiprep_dir, sub, ses, space, kind):
    """Find <qsiprep>/sub/(ses/)anat/*<kind>.nii.gz in the requested space; prefer this session's file, else the subject-level one."""
    if not (qsiprep_dir / sub).is_dir(): return None
    if space == "ACPC": ok = lambda p: "space-ACPC" in p.name
    else: ok = lambda p: "space-" not in p.name or f"space-{space}" in p.name  # native T1w carries no space entity
    cands = [p for p in (qsiprep_dir / sub).rglob(f"*{kind}.nii.gz") if p.parent.name == "anat" and ok(p)]
    mine = [p for p in cands if ses in p.parts or f"_{ses}_" in p.name]
    pick = mine or cands
    return sorted(pick)[0] if pick else None

def find_raw_t1w(raw_dir, sub, ses):
    """Raw BIDS T1w (native space = QSIPrep's space-T1w): <raw>/sub/ses/anat/*_T1w.nii.gz, else <raw>/sub/anat/."""
    cands = [p for p in (raw_dir / sub).rglob("*_T1w.nii.gz") if p.parent.name == "anat"]
    mine = [p for p in cands if ses in p.parts]
    pick = mine or cands
    return sorted(pick)[0] if pick else None

def density_and_outside(points, inv_affine, shape, mask):
    """Voxel-count map of the points on the anatomy grid, and the fraction of points outside the mask (or the image)."""
    vox = np.rint(nib.affines.apply_affine(inv_affine, points)).astype(int)
    inside = np.all((vox >= 0) & (vox < np.array(shape)), axis=1)
    v = vox[inside]
    dens = np.bincount(np.ravel_multi_index(v.T, shape), minlength=int(np.prod(shape))).reshape(shape).astype(np.float32)
    in_mask = int((mask[tuple(v.T)] > 0).sum()) if mask is not None else int(inside.sum())
    return dens, 1.0 - in_mask / max(1, len(points))

def to_ras(arr, affine):
    return nib.as_closest_canonical(nib.Nifti1Image(arr, affine)).get_fdata()

def make_figure(anat_ras, rows, title, out_png):
    """rows: [(label, density_ras)]; one row per entry with axial / coronal / sagittal slices through the density centroid."""
    lo, hi = np.percentile(anat_ras[anat_ras > 0], [1, 99]) if (anat_ras > 0).any() else (0, 1)
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 2.6 * len(rows)), squeeze=False)
    for r, (label, dens) in enumerate(rows):
        idx = np.argwhere(dens > 0)
        c = idx.mean(axis=0).astype(int) if len(idx) else np.array(anat_ras.shape) // 2
        views = [(lambda a: a[:, :, c[2]].T), (lambda a: a[:, c[1], :].T), (lambda a: a[c[0], :, :].T)]
        for k, view in enumerate(views):
            ax = axes[r, k]
            ax.imshow(view(anat_ras), cmap="gray", vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
            d = view(dens)
            ax.imshow(np.ma.masked_where(d <= 0, np.log1p(d)), cmap="jet", vmin=0, vmax=max(np.log1p(dens.max()), 1e-6), alpha=0.85, origin="lower", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if k == 0: ax.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

def qc_session(mode, sub, ses, dwi_dir, fig_dir, key_bundles):
    cfg = MODES[mode]
    pattern = f"_space-{cfg['space']}_model-gqi_bundle-*_streamlines.tck.gz"
    files = {f.name: f for f in dwi_dir.glob("*" + pattern)}
    for extra in EXTRA_BUNDLE_DIRS:
        for f in (QSIRECON_DIR / "derivatives" / extra).rglob(f"{sub}_{ses}_*" + pattern):
            files.setdefault(f.name, f)  # a file also present under qsirecon-DSIStudio is used from there
    files = [files[k] for k in sorted(files)]
    summary = dict(subject=sub, session=ses, space=cfg["space"], n_bundles=len(files), anat="", mask="", flags="")
    bundle_rows, flags = [], []
    if not files:
        summary["flags"] = f"NO_BUNDLES_IN_SPACE_{cfg['space']}"
        return summary, bundle_rows, set()

    if cfg["anat_kinds"] is None:  # moms: raw T1w, and no T1w-space brain mask exists
        anat_path, mask_path = find_raw_t1w(RAW_BIDS_DIR, sub, ses), None
    else:
        anat_path = next((p for k in cfg["anat_kinds"] if (p := find_anat(cfg["qsiprep"], sub, ses, cfg["space"], k))), None)
        mask_path = find_anat(cfg["qsiprep"], sub, ses, cfg["space"], "desc-brain_mask")
    summary["anat"], summary["mask"] = str(anat_path or ""), str(mask_path or "")
    if anat_path is None: flags.append("NO_ANAT")
    if mask_path is None and cfg["anat_kinds"] is not None: flags.append("NO_MASK")

    anat_img = nib.load(anat_path) if anat_path else None
    mask = np.asarray(nib.load(mask_path).dataobj) if mask_path else None
    if anat_img is not None and mask is not None and mask.shape != anat_img.shape[:3]:
        flags.append("MASK_GRID_MISMATCH")
        mask = None
    inv, shape = (np.linalg.inv(anat_img.affine), anat_img.shape[:3]) if anat_img is not None else (None, None)

    total = np.zeros(shape, dtype=np.float32) if shape else None
    key_dens, names = [], set()
    for f in files:
        name = f.name.split("_bundle-")[1].replace("_streamlines.tck.gz", "")
        names.add(name)
        n = tck_count(f)
        row = dict(subject=sub, session=ses, bundle=name, n_streamlines=n, frac_outside_mask="")
        if not n: flags.append(f"EMPTY:{name}")
        elif inv is not None:
            dens, outside = density_and_outside(load_points(f), inv, shape, mask)
            total += dens
            if mask is not None:
                row["frac_outside_mask"] = round(outside, 4)
                if outside > MAX_OUTSIDE_MASK: flags.append(f"OUTSIDE_MASK:{name}")
            if any(k in name for k in key_bundles): key_dens.append((name, dens))
        bundle_rows.append(row)

    summary["total_streamlines"] = sum(r["n_streamlines"] or 0 for r in bundle_rows)
    if anat_img is not None:
        affine = anat_img.affine
        data = np.asarray(anat_img.dataobj, dtype=np.float32)
        anat_ras = to_ras(data[..., 0] if data.ndim == 4 else data, affine)
        rows = [("ALL bundles", to_ras(total, affine))] + [(n, to_ras(d, affine)) for n, d in key_dens]
        out_png = fig_dir / f"{sub}_{ses}_{cfg['space']}_bundleQC.png"
        make_figure(anat_ras, rows, f"{sub} {ses} | {cfg['space']} | {anat_path.name}", out_png)
    summary["flags"] = ";".join(flags)
    return summary, bundle_rows, names

def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser(description="QC QSIRecon DSI Studio bundles (.tck.gz) over the QSIPrep anatomy.")
    ap.add_argument("--mode", required=True, choices=MODES, help="mom = space-T1w bundles, baby = space-ACPC bundles")
    ap.add_argument("--subjects", nargs="*", help="subject IDs (with or without 'sub-'); default: every subject of this cohort found in the outputs")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"QC output root (default {DEFAULT_OUT})")
    ap.add_argument("--qsirecon-dir", type=Path, help=f"QSIRecon output root containing derivatives/ (default {OUTPUT_DERIVATIVES_DIR})")
    ap.add_argument("--qsiprep-dir", type=Path, help="QSIPrep output root for this mode (babies: ACPC T2w/T1w + brain mask)")
    ap.add_argument("--rawdata-dir", type=Path, help=f"raw BIDS root, used for the moms' T1w (default {BASE_BIDS_DIR})")
    ap.add_argument("--limit", type=int, help="only QC the first N sessions (for testing)")
    ap.add_argument("--key-bundles", nargs="*", default=KEY_BUNDLES, help="bundle-name substrings to show as their own figure rows")
    args = ap.parse_args()

    global QSIRECON_DIR, RAW_BIDS_DIR
    cfg = MODES[args.mode]
    if args.qsirecon_dir: QSIRECON_DIR = args.qsirecon_dir
    if args.qsiprep_dir: cfg["qsiprep"] = args.qsiprep_dir
    if args.rawdata_dir: RAW_BIDS_DIR = args.rawdata_dir
    dsi_dir = QSIRECON_DIR / "derivatives" / CHECK_OUTPUT_DIR
    if not dsi_dir.is_dir(): sys.exit(f"[ERROR] {dsi_dir} not found")
    if args.subjects: subs = [s if s.startswith("sub-") else f"sub-{s}" for s in args.subjects]
    else: subs = sorted(d.name for d in dsi_dir.glob("sub-*") if d.is_dir() and cfg["is_cohort"](d.name[4:]))

    out_dir = args.out / f"qsirecon_bundles_{args.mode}"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"[QC] mode={args.mode} | space={cfg['space']} | {len(subs)} subjects | output: {out_dir}")

    summaries, bundle_rows, names_by_session = [], [], {}
    for sub in subs:
        sessions = sorted(d for d in (dsi_dir / sub).glob("ses-*") if (d / "dwi").is_dir())
        if not sessions:
            summaries.append(dict(subject=sub, session="", space=cfg["space"], n_bundles=0, flags="NO_SESSION_DWI_DIR"))
            print(f"  {sub}: no ses-*/dwi folder")
            continue
        for ses_dir in sessions:
            if args.limit and len(names_by_session) >= args.limit: break
            print(f"  {sub} {ses_dir.name} ...", flush=True)
            s, b, names = qc_session(args.mode, sub, ses_dir.name, ses_dir / "dwi", fig_dir, args.key_bundles)
            summaries.append(s); bundle_rows += b; names_by_session[(sub, ses_dir.name)] = names

    # bundles present in at least half of the sessions but absent from a given one
    common = {n for n in set().union(*names_by_session.values()) if sum(n in v for v in names_by_session.values()) >= len(names_by_session) / 2} if names_by_session else set()
    for s in summaries:
        names = names_by_session.get((s["subject"], s["session"]))
        if names is not None and common - names:
            s["flags"] = ";".join(filter(None, [s.get("flags", ""), f"MISSING_VS_COHORT:{len(common - names)}"]))
            s["missing_vs_cohort"] = " ".join(sorted(common - names))

    write_csv(out_dir / "qc_summary.csv", summaries, ["subject", "session", "space", "n_bundles", "total_streamlines", "anat", "mask", "flags", "missing_vs_cohort"])
    write_csv(out_dir / "qc_bundles.csv", bundle_rows, ["subject", "session", "bundle", "n_streamlines", "frac_outside_mask"])
    flagged = [s for s in summaries if s.get("flags")]
    print(f"\n====== BUNDLE QC ({args.mode.upper()}) ======")
    print(f"Sessions QC'd: {len(summaries)} | Flagged: {len(flagged)}")
    for s in flagged: print(f"  {s['subject']} {s['session']}: {s['flags']}")
    print(f"Results: {out_dir}")

if __name__ == "__main__":
    main()
