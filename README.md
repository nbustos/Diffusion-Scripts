# Diffusion-Scripts

Scripts for diffusion MRI processing.

- `combo_qsirecon.py` - QSIRecon combo script
- `combo_qsiprep.py` - interactive QSIPrep 1.0.0 launcher for COMBO moms and babies
- `qc_qsirecon_bundles.py` - QC of QSIRecon DSI Studio bundles (`--mode mom|baby`)
- `anx_cbt_tractography.py` - extract bilateral MRtrix Uncinate Fasciculus
  streamlines from each subject's `sift_1mio.tck`

## ANX-CBT uncinate tractography

The ANX-CBT extraction filters the existing SIFT-filtered whole-brain
tractogram; it does not generate new streamlines. Each subject must have
subject-space endpoint ROIs in the same world coordinates as its tractogram:

- `UF_L_frontal` and `UF_L_temporal`
- `UF_R_frontal` and `UF_R_temporal`

ROIs may use `.mif`, `.mif.gz`, `.nii`, or `.nii.gz` and can be placed in the
tractogram directory or an adjacent `ROIs/` directory. Optional
`UF_L_exclude` and `UF_R_exclude` masks are also supported.

The script provides three mask-definition modes:

1. `existing` (default): use ROI masks already prepared per subject.
2. `spheres`: generate bilateral frontal and anterior-temporal endpoint masks
   from subject-specific RAS coordinates and radii in a CSV file.
3. `labels`: generate endpoint masks by unioning anatomical labels from a
   subject-registered label image. Labels must be separated with `;` in CSV.

No coordinates or atlas label values are hard-coded because those choices must
match the cohort's anatomical conventions. The recommended scientific workflow
is to define and visually check the masks on representative subjects before
processing the full cohort.

Example sphere configuration:

```csv
subject,side,frontal_x_mm,frontal_y_mm,frontal_z_mm,frontal_radius_mm,temporal_x_mm,temporal_y_mm,temporal_z_mm,temporal_radius_mm
subject001,L,-10,45,-5,8,-28,3,-18,10
subject001,R,10,45,-5,8,28,3,-18,10
```

The coordinates above are placeholders, not validated UF definitions. Values
are RAS/scanner-space millimetres and must be changed for the study. Optional
exclusion-sphere columns are `exclude_x_mm`, `exclude_y_mm`, `exclude_z_mm`,
and `exclude_radius_mm`.

Generate the masks and tractograms together:

```bash
python3 anx_cbt_tractography.py \
  --subjects-root /data/FitzgeraldLab/Analysis/DTI/Subjects \
  --mask-mode spheres \
  --mask-config uf_spheres.csv \
  --output-dir /data/FitzgeraldLab/Analysis/DTI/Tractography/UF
```

Add `--masks-only` to create and inspect the masks before running tractography.

The label mode expects this CSV shape:

```csv
subject,side,label_image,frontal_labels,temporal_labels,exclude_labels
subject001,L,aparc+aseg.mif,1003;1012,1016,3
subject001,R,aparc+aseg.mif,2003;2012,2016,3
```

Label numbers above are placeholders. `label_image` may be absolute or relative
to the subject directory, and `{subject}` substitution is supported.

The endpoint masks are applied with `tckedit -ends_only`. When exclusion masks
are supplied, a second `tckedit` pass applies them to the full streamline path.

Audit the complete cohort without writing tractograms:

```bash
python3 anx_cbt_tractography.py \
  --subjects-root /data/FitzgeraldLab/Analysis/DTI/Subjects \
  --dry-run
```

Run extraction:

```bash
python3 anx_cbt_tractography.py \
  --subjects-root /data/FitzgeraldLab/Analysis/DTI/Subjects \
  --output-dir /data/FitzgeraldLab/Analysis/DTI/Tractography/UF
```

Outputs are separate `.tck` files for left and right UF plus
`uf_tractography_summary.csv`. Use `--roi-root` if the subject ROI folders are
stored elsewhere, or `--subjects ID1 ID2` for a subset.
