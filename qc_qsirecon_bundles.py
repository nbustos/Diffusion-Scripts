# qc_qsirecon_bundles.py - QC of QSIRecon DSI Studio bundles (MRtrix .tck.gz) for the COMBO cohorts. Run with --mode mom or --mode baby.
# Moms: space-ACPC bundles on the ACPC QSIPrep T1w. Babies: space-ACPC bundles on the ACPC QSIPrep T2w (T1w if present). Both use the ACPC brain mask.
# Per subject/session: streamline counts (read from .tck headers), fraction of streamline points outside the brain mask, and a PNG of bundle density on the anatomy.
# Flags empty bundles, bundles missing relative to the rest of the cohort, and bundles outside the brain mask. Writes CSVs + PNGs to a separate QC folder (inputs are read-only).
# Visual check: --view SUBJECT opens that subject's anatomy + mask + bundles in MRtrix mrview. Usage: python3 qc_qsirecon_bundles.py --mode mom [--subjects 1755138P R1762041 ...] [--out DIR] [--limit N]; needs numpy, nibabel, matplotlib (no MRtrix needed).


import argparse
import csv
import gzip
import re
import subprocess
import warnings
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.streamlines import TckFile
warnings.filterwarnings("ignore", message=".*encountered in matmul")  # spurious numpy-on-macOS warning; results verified identical
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from combo_qsirecon import INPUT_QSIPREP_MOMS_DIR, INPUT_QSIPREP_BABIES_DIR, OUTPUT_DERIVATIVES_DIR, CHECK_OUTPUT_DIR

MODES = {
    # space: the "space-" entity in the bundle filenames; anat_kinds: QSIPrep anatomical images to try, in order
    "mom":  dict(space="ACPC", qsiprep=INPUT_QSIPREP_MOMS_DIR,   anat_kinds=["desc-preproc_T1w"],
                 is_cohort=lambda i: i.lower().endswith("p")),
    "baby": dict(space="ACPC", qsiprep=INPUT_QSIPREP_BABIES_DIR, anat_kinds=["desc-preproc_T1w", "desc-preproc_T2w"],
                 is_cohort=lambda i: i[-1].isdigit()),
}
QSIRECON_DIR = OUTPUT_DERIVATIVES_DIR  # overridable with --qsirecon-dir
FLAT_BUNDLES_DIR = None  # --bundles-dir: one flat folder of <sub>_<ses>_*.tck.gz files instead of the QSIRecon tree
DEFAULT_OUT = Path("/Marshlab/data/COMBO/diffusion_qc")
# Local copy of the data (R_SYNC.sh layout): flat bundles + ANAT (QSIPrep anatomy) folders. If it exists it becomes the default,
# so e.g. `--view 1611701P` works with no path flags; on the EC2 the QSIRecon/QSIPrep paths above are used. Flags override.
LOCAL_ZI = Path("/Users/bustos/Documents/Projects/COMBO/DATA/Tractography/ZI")
if LOCAL_ZI.is_dir():
    FLAT_BUNDLES_DIR = LOCAL_ZI / "bundles"
    DEFAULT_OUT = LOCAL_ZI / "QC"
    for _cfg in MODES.values(): _cfg["qsiprep"] = LOCAL_ZI / "ANAT"
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

def find_anat(qsiprep_dir, sub, ses, space, kind, ses_only=False):
    """Find <qsiprep>/sub/(ses/)anat/*<kind>.nii.gz in the requested space; prefer this session's file, else (unless ses_only) the subject-level one."""
    if not (qsiprep_dir / sub).is_dir(): return None
    if space == "ACPC": ok = lambda p: "space-ACPC" in p.name
    else: ok = lambda p: "space-" not in p.name or f"space-{space}" in p.name  # native T1w carries no space entity
    cands = [p for p in (qsiprep_dir / sub).rglob(f"*{kind}.nii.gz") if p.parent.name == "anat" and ok(p)]
    mine = [p for p in cands if ses in p.parts or f"_{ses}_" in p.name]
    pick = mine if ses_only else (mine or cands)
    return sorted(pick)[0] if pick else None

def pick_anat(qsiprep_dir, sub, ses, space, kinds):
    """First anatomy found: session-level files (any kind, in order) before subject-level ones."""
    for ses_only in (True, False):
        for k in kinds:
            p = find_anat(qsiprep_dir, sub, ses, space, k, ses_only)
            if p: return p
    return None

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

def find_bundle_files(mode, sub, ses, dwi_dir):
    """Bundle .tck.gz files of one session in the mode's space: dwi folder, flat --bundles-dir, and the extra UF_* folders."""
    pattern = f"_space-{MODES[mode]['space']}_model-gqi_bundle-*_streamlines.tck.gz"
    files = {f.name: f for f in dwi_dir.glob("*" + pattern)} if dwi_dir else {}
    if FLAT_BUNDLES_DIR:
        files.update({f.name: f for f in FLAT_BUNDLES_DIR.glob(f"{sub}_{ses}*" + pattern) if f.name.startswith(f"{sub}_{ses}_")})
    for extra in ([] if FLAT_BUNDLES_DIR else EXTRA_BUNDLE_DIRS):
        for f in (QSIRECON_DIR / "derivatives" / extra).rglob(f"{sub}_{ses}*" + pattern):
            files.setdefault(f.name, f)  # a file also present under qsirecon-DSIStudio is used from there
    return [files[k] for k in sorted(files)]

def find_anat_and_mask(mode, sub, ses):
    cfg = MODES[mode]
    anat_path = pick_anat(cfg["qsiprep"], sub, ses, cfg["space"], cfg["anat_kinds"])
    # the mask must come from the same folder as the anatomy (same grid)
    masks = sorted(p for p in anat_path.parent.glob("*desc-brain_mask.nii.gz") if f"space-{cfg['space']}" in p.name) if anat_path else []
    return anat_path, (masks[0] if masks else None)

def list_sessions(sub):
    """[(session, dwi dir or None)] of a subject, from the flat bundles folder or the QSIRecon tree."""
    if FLAT_BUNDLES_DIR:
        return sorted({(m.group(1), None) for f in FLAT_BUNDLES_DIR.glob(f"{sub}_ses-*.tck.gz") if (m := re.match(rf"{re.escape(sub)}_(ses-[^_]+)_", f.name))})
    sub_dir = QSIRECON_DIR / "derivatives" / CHECK_OUTPUT_DIR / sub
    return [(d.name, d / "dwi") for d in sorted(sub_dir.glob("ses-*")) if (d / "dwi").is_dir()]

def view_in_mrview(mode, sub, session, only, dry_run, mask_opacity):
    """Open a subject's anatomy, brain mask and bundles together in MRtrix's mrview for visual QC."""
    sessions = list_sessions(sub)
    if not sessions: sys.exit(f"[ERROR] no sessions with bundles found for {sub}")
    if session:
        sessions = [x for x in sessions if x[0] == session]
        if not sessions: sys.exit(f"[ERROR] {sub} has no session {session}")
    if len(sessions) > 1: sys.exit(f"[ERROR] {sub} has several sessions: {', '.join(x[0] for x in sessions)}. Pick one with --session.")
    ses, dwi_dir = sessions[0]
    files = [f for f in find_bundle_files(mode, sub, ses, dwi_dir) if not only or any(o in f.name for o in only)]
    anat, mask = find_anat_and_mask(mode, sub, ses)
    if not anat: sys.exit(f"[ERROR] no {MODES[mode]['space']} anatomy found for {sub} {ses} under {MODES[mode]['qsiprep']}")
    if not files: sys.exit(f"[ERROR] no {MODES[mode]['space']} bundle files found for {sub} {ses}")
    print(f"[VIEW] {sub} {ses} | mode={mode}\n  anat: {anat}\n  mask: {mask}\n  bundles ({len(files)}):")
    for f in files: print(f"    {f.name}")
    if not dry_run and not shutil.which("mrview"): sys.exit("[ERROR] mrview not found on PATH (install MRtrix3)")
    with tempfile.TemporaryDirectory() as tmp:  # mrview cannot read .tck.gz, so unzip into a temp folder
        tcks = []
        for f in files:
            out = Path(tmp) / f.name[:-3]
            if not dry_run:
                with gzip.open(f, "rb") as src, open(out, "wb") as dst: shutil.copyfileobj(src, dst)
            tcks.append(str(out))
        cmd = ["mrview", str(anat), "-mode", "2"]  # 2 = ortho view (mrview otherwise reopens in whichever mode was used last)
        if mask: cmd += ["-roi.load", str(mask), "-roi.opacity", str(mask_opacity), "-roi.colour", "0,1,0"]
        for t in tcks: cmd += ["-tractography.load", t]
        print("\n" + " ".join(cmd))
        if not dry_run: subprocess.run(cmd)  # blocks until mrview is closed, then the temp files are removed

COLLATE_FIELDS = ["subject_id", "session_id", "Mom", "mean_fd", "raw_num_bad_slices", "bad_slice_fraction",
                  "raw_masked_neighbor_corr", "raw_num_directions", "raw_max_b", "t1_dice_distance"]

def collate_motion(motion_dir, out_csv):
    """One row per QSIPrep *desc-image_qc.tsv: the five usability metrics + Mom (1 = mom, 0 = baby; moms' IDs end in P)."""
    num = lambda r, c: float(r[c]) if r.get(c) not in (None, "", "n/a") else None
    rows = []
    for f in sorted(motion_dir.glob("*desc-image_qc.tsv")):
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                nz, nd, nb = num(r, "raw_dimension_z"), num(r, "raw_num_directions"), num(r, "raw_num_bad_slices")
                sid = r.get("subject_id") or f.name.split("_")[0]
                rows.append(dict(
                    subject_id=sid, session_id=r.get("session_id") or f.name.split("_")[1],
                    Mom=int(MODES["mom"]["is_cohort"](sid.removeprefix("sub-"))),
                    mean_fd=num(r, "mean_fd"), raw_num_bad_slices=nb,
                    bad_slice_fraction=round(nb / (nz * nd), 6) if nb is not None and nz and nd else None,
                    raw_masked_neighbor_corr=num(r, "raw_masked_neighbor_corr"), raw_num_directions=nd,
                    raw_max_b=num(r, "raw_max_b"), t1_dice_distance=num(r, "t1_dice_distance")))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_csv, rows, COLLATE_FIELDS)
    print(f"[COLLATE] {len(rows)} sessions ({sum(r['Mom'] for r in rows)} moms, {sum(1 - r['Mom'] for r in rows)} babies) from {motion_dir}\n  -> {out_csv}")

def qc_session(mode, sub, ses, dwi_dir, fig_dir, key_bundles):
    cfg = MODES[mode]
    files = find_bundle_files(mode, sub, ses, dwi_dir)
    summary = dict(subject=sub, session=ses, space=cfg["space"], n_bundles=len(files), anat="", mask="", flags="")
    bundle_rows, flags = [], []
    if not files:
        summary["flags"] = f"NO_BUNDLES_IN_SPACE_{cfg['space']}"
        return summary, bundle_rows, set()

    anat_path, mask_path = find_anat_and_mask(mode, sub, ses)
    summary["anat"], summary["mask"] = str(anat_path or ""), str(mask_path or "")
    if anat_path is None: flags.append("NO_ANAT")
    if mask_path is None: flags.append("NO_MASK")

    anat_img, anat_data, mask = None, None, None
    try:  # a truncated/partly synced file must not stop the whole run
        if anat_path:
            anat_img = nib.load(anat_path)
            anat_data = np.asarray(anat_img.dataobj, dtype=np.float32)
        mask = np.asarray(nib.load(mask_path).dataobj) if mask_path else None
    except Exception as e:
        flags.append(f"ANAT_UNREADABLE:{type(e).__name__}")
        anat_img, anat_data, mask = None, None, None
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
        anat_ras = to_ras(anat_data[..., 0] if anat_data.ndim == 4 else anat_data, affine)
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
    ap.add_argument("--mode", choices=MODES, help="mom or baby (both use space-ACPC bundles); optional with --view, where it is guessed from the ID (...P = mom)")
    ap.add_argument("--subjects", nargs="*", help="subject IDs (with or without 'sub-'); default: every subject of this cohort found in the outputs")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"QC output root (default {DEFAULT_OUT})")
    ap.add_argument("--qsirecon-dir", type=Path, help=f"QSIRecon output root containing derivatives/ (default {OUTPUT_DERIVATIVES_DIR})")
    ap.add_argument("--bundles-dir", type=Path, help="flat folder of <sub>_<ses>_*_bundle-*.tck.gz files (e.g. ZI/bundles); replaces the QSIRecon tree")
    ap.add_argument("--qsiprep-dir", type=Path, help="QSIPrep output root (or a folder with the synced sub-*/anat files) holding the ACPC anatomy + brain mask")
    ap.add_argument("--view", metavar="SUBJECT", help="open this subject's anatomy + brain mask + bundles together in MRtrix mrview, then exit (no QC run)")
    ap.add_argument("--session", help="with --view: session to open (e.g. ses-44445055); needed if the subject has several")
    ap.add_argument("--only", nargs="*", help="with --view: load only bundles whose name contains one of these (e.g. Uncinate)")
    ap.add_argument("--mask-opacity", type=float, default=0.1, help="with --view: brain mask overlay opacity, 0-1 (default 0.1)")
    ap.add_argument("--dry-run", action="store_true", help="with --view: print what would be opened and the mrview command, without launching it")
    ap.add_argument("--collate", action="store_true", help="write one CSV of the motion/usability metrics for all subjects (+ Mom column: 1 = mom, 0 = baby) from the QSIPrep *desc-image_qc.tsv files, then exit")
    ap.add_argument("--motion-dir", type=Path, default=LOCAL_ZI / "motion", help="with --collate: folder of *desc-image_qc.tsv files (default: ZI/motion)")
    ap.add_argument("--limit", type=int, help="only QC the first N sessions (for testing)")
    ap.add_argument("--key-bundles", nargs="*", default=KEY_BUNDLES, help="bundle-name substrings to show as their own figure rows")
    args = ap.parse_args()

    global QSIRECON_DIR, FLAT_BUNDLES_DIR
    if args.collate:
        collate_motion(args.motion_dir, args.out / "motion_qc_metrics.csv")
        return
    if not args.mode:
        if not args.view: ap.error("--mode is required (unless using --view)")
        args.mode = "mom" if args.view.lower().endswith("p") else "baby"
    cfg = MODES[args.mode]
    if args.qsirecon_dir: QSIRECON_DIR = args.qsirecon_dir
    if args.bundles_dir: FLAT_BUNDLES_DIR = args.bundles_dir
    elif args.qsirecon_dir: FLAT_BUNDLES_DIR = None  # an explicit QSIRecon tree overrides the local flat-folder default
    if args.qsiprep_dir: cfg["qsiprep"] = args.qsiprep_dir
    if args.view:
        view_in_mrview(args.mode, args.view if args.view.startswith("sub-") else f"sub-{args.view}", args.session, args.only, args.dry_run, args.mask_opacity)
        return
    # sessions to QC: (subject, session, dwi dir or None for a flat bundles folder)
    jobs, summaries = [], []
    if FLAT_BUNDLES_DIR:
        found = sorted({m.groups() for f in FLAT_BUNDLES_DIR.glob("sub-*_ses-*.tck.gz") if (m := re.match(r"(sub-[^_]+)_(ses-[^_]+)_", f.name))})
        wanted = {s if s.startswith("sub-") else f"sub-{s}" for s in args.subjects} if args.subjects else None
        jobs = [(sub, ses, None) for sub, ses in found if (sub in wanted if wanted else cfg["is_cohort"](sub[4:]))]
        subs = sorted({j[0] for j in jobs})
    else:
        dsi_dir = QSIRECON_DIR / "derivatives" / CHECK_OUTPUT_DIR
        if not dsi_dir.is_dir(): sys.exit(f"[ERROR] {dsi_dir} not found")
        if args.subjects: subs = [s if s.startswith("sub-") else f"sub-{s}" for s in args.subjects]
        else: subs = sorted(d.name for d in dsi_dir.glob("sub-*") if d.is_dir() and cfg["is_cohort"](d.name[4:]))
        for sub in subs:
            sessions = sorted(d for d in (dsi_dir / sub).glob("ses-*") if (d / "dwi").is_dir())
            if not sessions:
                summaries.append(dict(subject=sub, session="", space=cfg["space"], n_bundles=0, flags="NO_SESSION_DWI_DIR"))
                print(f"  {sub}: no ses-*/dwi folder")
            jobs += [(sub, d.name, d / "dwi") for d in sessions]

    out_dir = args.out / f"qsirecon_bundles_{args.mode}"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"[QC] mode={args.mode} | space={cfg['space']} | {len(subs)} subjects, {len(jobs)} sessions | output: {out_dir}")

    bundle_rows, names_by_session = [], {}
    for sub, ses, dwi_dir in jobs[:args.limit]:
        print(f"  {sub} {ses} ...", flush=True)
        s, b, names = qc_session(args.mode, sub, ses, dwi_dir, fig_dir, args.key_bundles)
        summaries.append(s); bundle_rows += b; names_by_session[(sub, ses)] = names

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
