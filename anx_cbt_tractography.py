#!/usr/bin/env python3
"""Extract bilateral uncinate fasciculus streamlines from SIFT-filtered tractograms."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


SIDES = ("L", "R")
IMAGE_SUFFIXES = (".mif", ".mif.gz", ".nii", ".nii.gz")
SPHERE_COLUMNS = {
    "subject",
    "side",
    "frontal_x_mm",
    "frontal_y_mm",
    "frontal_z_mm",
    "frontal_radius_mm",
    "temporal_x_mm",
    "temporal_y_mm",
    "temporal_z_mm",
    "temporal_radius_mm",
}
LABEL_COLUMNS = {
    "subject",
    "side",
    "label_image",
    "frontal_labels",
    "temporal_labels",
}


@dataclass(frozen=True)
class RoiSet:
    frontal: Path
    temporal: Path
    exclusions: tuple[Path, ...]


@dataclass(frozen=True)
class Job:
    subject: str
    tractogram: Path
    side: str
    rois: RoiSet
    output: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract left and right Uncinate Fasciculus bundles from MRtrix "
            "sift_1mio.tck files using subject-space endpoint ROIs."
        )
    )
    parser.add_argument(
        "--subjects-root",
        type=Path,
        default=Path(
            os.environ.get(
                "ANX_CBT_SUBJECTS_ROOT",
                "/data/FitzgeraldLab/Analysis/DTI/Subjects",
            )
        ),
        help="directory whose child directories are subject/session folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: <subjects-root>/UF_tractography)",
    )
    parser.add_argument(
        "--roi-root",
        type=Path,
        help="optional root for per-subject ROIs (default: search each tractogram directory)",
    )
    parser.add_argument(
        "--mask-mode",
        choices=("existing", "spheres", "labels"),
        default="existing",
        help=(
            "existing: use saved subject-space masks; spheres: generate endpoint "
            "masks from RAS coordinates; labels: combine anatomical image labels"
        ),
    )
    parser.add_argument(
        "--mask-config",
        type=Path,
        help="CSV mask definitions; required for --mask-mode spheres or labels",
    )
    parser.add_argument(
        "--reference-name",
        default="mask.mif",
        help="per-tractogram reference image used for spherical masks (default: mask.mif)",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        metavar="SUBJECT",
        help="only process these immediate subject-directory names",
    )
    parser.add_argument(
        "--tck-name",
        default="sift_1mio.tck",
        help="input tractogram filename (default: sift_1mio.tck)",
    )
    parser.add_argument(
        "--minimum-streamlines",
        type=int,
        default=1,
        help="fail a completed extraction if fewer streamlines survive (default: 1)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing bundle outputs"
    )
    parser.add_argument(
        "--masks-only",
        action="store_true",
        help="generate masks and stop before running tckedit on tractograms",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="validate inputs and print commands only"
    )
    return parser.parse_args()


def subject_directories(root: Path, requested: list[str] | None) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"subjects root does not exist: {root}")
    dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if requested:
        available = {path.name: path for path in dirs}
        missing = sorted(set(requested) - available.keys())
        if missing:
            raise ValueError(
                "requested subjects are not immediate children of "
                f"{root}: {', '.join(missing)}"
            )
        dirs = [available[name] for name in requested]
    return dirs


def canonical_roi_names(side: str) -> dict[str, tuple[str, ...]]:
    tract = f"Uncinate_Fasciculus_{side}"
    lower_tract = tract.lower()
    return {
        "frontal": (
            f"UF_{side}_frontal",
            f"{tract}_frontal",
            f"{lower_tract}_frontal",
            f"UF_{side}_frontal_roi",
        ),
        "temporal": (
            f"UF_{side}_temporal",
            f"{tract}_temporal",
            f"{lower_tract}_temporal",
            f"UF_{side}_anterior_temporal",
            f"UF_{side}_temporal_roi",
        ),
        "exclude": (
            f"UF_{side}_exclude",
            f"UF_{side}_midline_exclude",
            f"UF_{side}_exclusions",
        ),
    }


def image_paths(directory: Path, stems: tuple[str, ...]) -> list[Path]:
    stem_lookup = {stem.lower() for stem in stems}
    matches = []
    if not directory.is_dir():
        return matches
    for path in directory.iterdir():
        name = path.name.lower()
        stem = next(
            (name[: -len(suffix)] for suffix in IMAGE_SUFFIXES if name.endswith(suffix)),
            None,
        )
        if stem in stem_lookup:
            matches.append(path)
    return sorted(matches)


def roi_search_directories(
    tractogram: Path, subjects_root: Path, roi_root: Path | None, subject: str
) -> list[Path]:
    tract_dir = tractogram.parent
    candidates = [tract_dir / "ROIs", tract_dir / "rois", tract_dir]
    if roi_root:
        candidates += [
            roi_root / subject / tract_dir.name / "ROIs",
            roi_root / subject / tract_dir.name,
            roi_root / subject / "ROIs",
            roi_root / subject,
        ]
    else:
        candidates += [
            subjects_root / subject / "ROIs",
            subjects_root / subject,
        ]
    unique = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def find_roi(
    name_stems: tuple[str, ...],
    directories: list[Path],
    required: bool,
) -> Path | None:
    for directory in directories:
        matches = image_paths(directory, name_stems)
        if len(matches) > 1:
            rendered = ", ".join(str(path) for path in matches)
            raise ValueError(f"multiple ROI files match {name_stems[0]}: {rendered}")
        if matches:
            return matches[0]
    if required:
        raise ValueError(f"missing required ROI: {name_stems[0]}")
    return None


def find_rois(
    tractogram: Path,
    subjects_root: Path,
    roi_root: Path | None,
    subject: str,
    side: str,
) -> RoiSet:
    names = canonical_roi_names(side)
    directories = roi_search_directories(tractogram, subjects_root, roi_root, subject)
    frontal = find_roi(names["frontal"], directories, required=True)
    temporal = find_roi(names["temporal"], directories, required=True)
    exclude = find_roi(names["exclude"], directories, required=False)
    assert frontal and temporal
    return RoiSet(frontal, temporal, (exclude,) if exclude else ())


def discover_jobs(
    args: argparse.Namespace, generated_rois: dict[tuple[str, str], RoiSet] | None = None
) -> tuple[list[Job], list[str]]:
    subjects = subject_directories(args.subjects_root, args.subjects)
    output_root = args.output_dir or (args.subjects_root / "UF_tractography")
    excluded_roots = {output_root.resolve()}
    if args.roi_root:
        excluded_roots.add(args.roi_root.resolve())
    subjects = [
        subject for subject in subjects if subject.resolve() not in excluded_roots
    ]
    jobs = []
    errors = []
    for subject_dir in subjects:
        tractograms = sorted(subject_dir.rglob(args.tck_name))
        if not tractograms:
            errors.append(f"{subject_dir.name}: no {args.tck_name} found")
            continue
        if len(tractograms) > 1:
            errors.append(
                f"{subject_dir.name}: found multiple {args.tck_name} files: "
                + ", ".join(str(path) for path in tractograms)
            )
            continue
        for tractogram in tractograms:
            for side in SIDES:
                output = output_root / subject_dir.name / (
                    f"{subject_dir.name}.Uncinate_Fasciculus_{side}.tck"
                )
                try:
                    rois = (
                        generated_rois.get((subject_dir.name, side))
                        if generated_rois
                        else None
                    )
                    if rois is None:
                        rois = find_rois(
                            tractogram,
                            args.subjects_root,
                            args.roi_root,
                            subject_dir.name,
                            side,
                        )
                    jobs.append(Job(subject_dir.name, tractogram, side, rois, output))
                except ValueError as exc:
                    errors.append(f"{subject_dir.name} [{side}]: {exc}")
    return jobs, errors


def read_mask_config(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"mask config does not exist: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(required - fields)
        if missing:
            raise ValueError(
                f"mask config {path} is missing column(s): {', '.join(missing)}"
            )
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def config_key(row: dict[str, str], config: Path) -> tuple[str, str]:
    subject = row["subject"]
    side = row["side"].upper()
    if not subject:
        raise ValueError(f"mask config {config} contains an empty subject")
    if side not in SIDES:
        raise ValueError(f"mask config {config}: side for {subject} must be L or R")
    return subject, side


def generated_roi_paths(root: Path, subject: str, side: str) -> RoiSet:
    folder = root / subject
    tract = f"UF_{side}"
    return RoiSet(
        folder / f"{tract}_frontal.nii.gz",
        folder / f"{tract}_temporal.nii.gz",
        (folder / f"{tract}_exclude.nii.gz",),
    )


def path_from_config(raw: str, subject_dir: Path, base: Path) -> Path:
    expanded = Path(raw.format(subject=subject_dir.name))
    if expanded.is_absolute():
        return expanded.resolve()
    subject_relative = subject_dir / expanded
    return (subject_relative if subject_relative.exists() else base / expanded).resolve()


def label_values(raw: str, field: str, subject: str, side: str) -> tuple[int, ...]:
    values = [part.strip() for part in re.split(r"[;|+]", raw) if part.strip()]
    try:
        labels = tuple(int(value) for value in values)
    except ValueError as exc:
        raise ValueError(
            f"invalid {field} labels for {subject} [{side}]: expected integers"
        ) from exc
    if not labels:
        raise ValueError(f"{field} is empty for {subject} [{side}]")
    return labels


def label_mask_command(image: Path, labels: tuple[int, ...], output: Path) -> list[str]:
    command = ["mrcalc"]
    for index, label in enumerate(labels):
        command += [str(image), str(label), "-eq"]
        if index:
            command += ["-add"]
    command += ["0", "-gt", str(output)]
    return command


def prepare_label_masks(
    args: argparse.Namespace, rows: list[dict[str, str]], roi_root: Path
) -> dict[tuple[str, str], RoiSet]:
    generated = {}
    for row in rows:
        subject, side = config_key(row, args.mask_config)
        subject_dir = args.subjects_root / subject
        if args.subjects and subject not in args.subjects:
            continue
        if not subject_dir.is_dir():
            raise ValueError(f"label config references unknown subject: {subject}")
        key = (subject, side)
        if key in generated:
            raise ValueError(f"duplicate mask definition for {subject} [{side}]")
        label_image = path_from_config(row["label_image"], subject_dir, args.mask_config.parent)
        if not label_image.is_file():
            raise ValueError(
                f"label image for {subject} [{side}] does not exist: {label_image}"
            )
        rois = generated_roi_paths(roi_root, subject, side)
        definitions = (
            ("frontal_labels", rois.frontal),
            ("temporal_labels", rois.temporal),
        )
        for field, output in definitions:
            labels = label_values(row[field], field, subject, side)
            run_output_command(
                label_mask_command(label_image, labels, output),
                output,
                args.overwrite,
                args.dry_run,
            )
        exclusions = ()
        if row.get("exclude_labels"):
            labels = label_values(
                row["exclude_labels"], "exclude_labels", subject, side
            )
            run_output_command(
                label_mask_command(label_image, labels, rois.exclusions[0]),
                rois.exclusions[0],
                args.overwrite,
                args.dry_run,
            )
            exclusions = rois.exclusions
        generated[key] = RoiSet(rois.frontal, rois.temporal, exclusions)
    return generated


def number(row: dict[str, str], column: str, subject: str, side: str) -> float:
    try:
        return float(row[column])
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"invalid numeric value in {column} for {subject} [{side}]"
        ) from exc


def sphere_array(reference: Path, sphere: tuple[float, float, float, float]) -> tuple:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "spherical masks require Python packages nibabel and numpy"
        ) from exc

    if reference.suffix.lower() == ".mif":
        with tempfile.TemporaryDirectory() as temporary:
            converted = Path(temporary) / "reference.nii.gz"
            subprocess.run(["mrconvert", str(reference), str(converted)], check=True)
            image = nib.load(str(converted))
            if len(image.shape) != 3:
                raise ValueError(f"spherical mask reference must be 3D: {reference}")
            affine = image.affine.copy()
            shape = image.shape
            header = image.header.copy()
    else:
        image = nib.load(str(reference))
        if len(image.shape) != 3:
            raise ValueError(f"spherical mask reference must be 3D: {reference}")
        affine = image.affine
        shape = image.shape
        header = image.header.copy()
    x, y, z, radius = sphere
    if radius <= 0:
        raise ValueError(f"sphere radius must be positive, got {radius}")
    coordinates = np.indices(shape, dtype=np.float32)
    world = nib.affines.apply_affine(affine, np.moveaxis(coordinates, 0, -1))
    center = np.asarray((x, y, z), dtype=np.float32)
    data = (np.sum((world - center) ** 2, axis=-1) <= radius**2).astype(np.uint8)
    return nib, affine, header, data


def write_sphere_mask(
    reference: Path, sphere: tuple[float, float, float, float], output: Path
) -> None:
    nib, affine, header, data = sphere_array(reference, sphere)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine, header), output)


def prepare_sphere_masks(
    args: argparse.Namespace, rows: list[dict[str, str]], roi_root: Path
) -> dict[tuple[str, str], RoiSet]:
    generated = {}
    for row in rows:
        subject, side = config_key(row, args.mask_config)
        if args.subjects and subject not in args.subjects:
            continue
        subject_dir = args.subjects_root / subject
        if not subject_dir.is_dir():
            raise ValueError(f"sphere config references unknown subject: {subject}")
        key = (subject, side)
        if key in generated:
            raise ValueError(f"duplicate mask definition for {subject} [{side}]")
        references = sorted(subject_dir.rglob(args.reference_name))
        if len(references) != 1:
            raise ValueError(
                f"{subject}: expected exactly one {args.reference_name}, "
                f"found {len(references)}"
            )
        spheres = {}
        for endpoint in ("frontal", "temporal"):
            spheres[endpoint] = tuple(
                number(row, f"{endpoint}_{suffix}", subject, side)
                for suffix in ("x_mm", "y_mm", "z_mm", "radius_mm")
            )
        rois = generated_roi_paths(roi_root, subject, side)
        for endpoint, output in (
            ("frontal", rois.frontal),
            ("temporal", rois.temporal),
        ):
            ensure_output(output, args.overwrite, args.dry_run)
            if args.dry_run:
                print(f"[MASK] sphere {spheres[endpoint]} -> {output}")
            else:
                write_sphere_mask(references[0], spheres[endpoint], output)
        exclusions = ()
        if all(
            row.get(f"exclude_{suffix}")
            for suffix in ("x_mm", "y_mm", "z_mm", "radius_mm")
        ):
            exclusion = tuple(
                number(row, f"exclude_{suffix}", subject, side)
                for suffix in ("x_mm", "y_mm", "z_mm", "radius_mm")
            )
            ensure_output(rois.exclusions[0], args.overwrite, args.dry_run)
            if args.dry_run:
                print(f"[MASK] sphere {exclusion} -> {rois.exclusions[0]}")
            else:
                write_sphere_mask(references[0], exclusion, rois.exclusions[0])
            exclusions = rois.exclusions
        generated[key] = RoiSet(rois.frontal, rois.temporal, exclusions)
    return generated


def ensure_output(output: Path, overwrite: bool, dry_run: bool) -> None:
    if output.exists() and not overwrite:
        raise ValueError(f"mask output exists (use --overwrite): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return
    output.unlink(missing_ok=True)


def run_output_command(
    command: list[str], output: Path, overwrite: bool, dry_run: bool
) -> None:
    ensure_output(output, overwrite, dry_run)
    print(f"[MASK] {shlex.join(command)}")
    if not dry_run:
        subprocess.run(command, check=True)


def prepare_masks(args: argparse.Namespace) -> dict[tuple[str, str], RoiSet] | None:
    if args.mask_mode == "existing":
        return None
    if not args.mask_config:
        raise ValueError("--mask-config is required for generated masks")
    args.mask_config = args.mask_config.resolve()
    roi_root = (args.roi_root or (args.subjects_root / "UF_ROIs")).resolve()
    args.roi_root = roi_root
    if args.mask_mode == "labels":
        return prepare_label_masks(
            args, read_mask_config(args.mask_config, LABEL_COLUMNS), roi_root
        )
    return prepare_sphere_masks(
        args, read_mask_config(args.mask_config, SPHERE_COLUMNS), roi_root
    )


def tckedit_commands(job: Job, threads: int = 0) -> list[list[str]]:
    intermediate = job.output.with_name(f".{job.output.stem}.endpoints.tck")
    endpoint_output = intermediate if job.rois.exclusions else job.output
    commands = [[
        "tckedit",
        str(job.tractogram),
        str(endpoint_output),
        "-include",
        str(job.rois.frontal),
        "-include",
        str(job.rois.temporal),
        "-ends_only",
        "-nthreads",
        str(threads),
    ]]
    if job.rois.exclusions:
        exclusion_command = [
            "tckedit",
            str(intermediate),
            str(job.output),
        ]
        for exclusion in job.rois.exclusions:
            exclusion_command += ["-exclude", str(exclusion)]
        exclusion_command += ["-nthreads", str(threads)]
        commands.append(exclusion_command)
    return commands


def streamline_count(path: Path) -> int:
    if not path.exists():
        raise RuntimeError(f"MRtrix did not create expected output: {path}")
    result = subprocess.run(
        ["tckinfo", str(path)], check=True, text=True, capture_output=True
    )
    match = re.search(r"^\s*count:\s*(\d+)\s*$", result.stdout, flags=re.MULTILINE)
    if not match:
        match = re.search(r"^\s*count:\s*(\d+)\s*$", result.stderr, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"could not read streamline count from: {path}")
    return int(match.group(1))


def require_tools(dry_run: bool) -> None:
    required = ("tckedit",) if dry_run else ("tckedit", "tckinfo")
    missing = [tool for tool in required if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"required MRtrix command(s) not found on PATH: {', '.join(missing)}")


def run(args: argparse.Namespace) -> int:
    args.subjects_root = args.subjects_root.resolve()
    if args.output_dir:
        args.output_dir = args.output_dir.resolve()
    if args.roi_root:
        args.roi_root = args.roi_root.resolve()
    if args.mask_mode == "labels":
        if not shutil.which("mrcalc"):
            raise RuntimeError("required MRtrix command not found on PATH: mrcalc")
    elif args.mask_mode == "spheres" and not shutil.which("mrconvert"):
        raise RuntimeError("required MRtrix command not found on PATH: mrconvert")
    generated_rois = prepare_masks(args)
    if args.masks_only:
        if args.mask_mode == "existing":
            raise ValueError("--masks-only requires --mask-mode spheres or labels")
        print(f"Generated {len(generated_rois or {})} mask definition(s).")
        return 0
    require_tools(args.dry_run)
    jobs, errors = discover_jobs(args, generated_rois)
    if errors:
        print("ROI/input audit failed:", file=sys.stderr)
        for error in errors:
            print(f"  ERROR: {error}", file=sys.stderr)
        if args.mask_mode == "existing":
            print(
                "\nExpected subject-space ROI names are UF_L_frontal, UF_L_temporal, "
                "UF_R_frontal, and UF_R_temporal with .mif/.nii/.nii.gz suffixes.",
                file=sys.stderr,
            )
        return 2
    if not jobs:
        print("No tractography jobs found.", file=sys.stderr)
        return 2

    output_root = args.output_dir or (args.subjects_root / "UF_tractography")
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "uf_tractography_summary.csv"
    rows = []
    failures = []
    for job in jobs:
        commands = tckedit_commands(job)
        intermediate = job.output.with_name(f".{job.output.stem}.endpoints.tck")
        print(f"[{job.subject} {job.side}]")
        for command in commands:
            print(f"  {shlex.join(command)}")
        try:
            if args.dry_run:
                continue
            if job.output.exists() and not args.overwrite:
                raise RuntimeError(f"output exists (use --overwrite): {job.output}")
            job.output.parent.mkdir(parents=True, exist_ok=True)
            intermediate.unlink(missing_ok=True)
            if args.overwrite:
                job.output.unlink(missing_ok=True)
            for command in commands:
                subprocess.run(command, check=True)
            count = streamline_count(job.output)
            if count < args.minimum_streamlines:
                raise RuntimeError(
                    f"only {count} streamlines survived; minimum is "
                    f"{args.minimum_streamlines}"
                )
            rows.append(
                {
                    "subject": job.subject,
                    "tract": f"Uncinate_Fasciculus_{job.side}",
                    "hemisphere": job.side,
                    "input_tractogram": str(job.tractogram),
                    "output_tractogram": str(job.output),
                    "frontal_roi": str(job.rois.frontal),
                    "temporal_roi": str(job.rois.temporal),
                    "exclusion_rois": ";".join(map(str, job.rois.exclusions)),
                    "streamlines": count,
                }
            )
            print(f"  wrote {job.output} ({count} streamlines)")
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            job.output.unlink(missing_ok=True)
            failures.append(f"{job.subject} [{job.side}]: {exc}")
        finally:
            intermediate.unlink(missing_ok=True)

    if rows and not args.dry_run:
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote summary: {summary_path}")
    if failures:
        print("Processing failed:", file=sys.stderr)
        for failure in failures:
            print(f"  ERROR: {failure}", file=sys.stderr)
        return 1
    print(f"Validated {len(jobs)} UF extraction jobs.")
    return 0


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
