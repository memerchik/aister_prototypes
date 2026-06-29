#!/usr/bin/env python3
"""Build a gallery/query image-retrieval dataset from ABO 360-degree spins.

The script downloads only matching Amazon Berkeley Objects (ABO) spin images.
For each object it:

1. Sorts all spin frames by azimuth.
2. Samples every Nth frame (default: every 6th frame).
3. Automatically reduces N for short spins so at least 8 gallery images remain.
4. Optionally limits gallery views with evenly spaced sampling.
5. Randomly holds out one distinct frame as the query image.
5. Writes the requested directory structure:

    OUTPUT/
      gallery/
        object_001/
          img_001.jpg
          img_002.jpg
      query/
        query_001.jpg
      query_labels.csv       # only with --write-query-labels

Example:
    pip install requests
    python3 download_abo_retrieval_dataset.py \
        --product-type SOFA \
        --output workshop_materials/data \
        --max-objects 10 \
        --max-gallery-images 4 \
        --write-query-labels
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import random
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

SCRIPT_VERSION = "3.0.0"
BASE_URL = "https://amazon-berkeley-objects.s3.amazonaws.com"
LISTING_SHARDS = tuple("0123456789abcdef")


@dataclass(frozen=True)
class ObjectPlan:
    """Download plan for one output object."""

    listing: dict[str, Any]
    source_count: int
    sampling_summary: str
    gallery_rows: list[dict[str, str]]
    query_row: dict[str, str]


def values_from_field(value: Any) -> list[str]:
    """Normalize ABO string/list/dictionary metadata fields to strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        normalized = value.get("value")
        return [str(normalized)] if normalized is not None else []
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if isinstance(item, dict):
                normalized = item.get("value")
                if normalized is not None:
                    result.append(str(normalized))
            elif item is not None:
                result.append(str(item))
        return result
    return [str(value)]


def english_titles(item: dict[str, Any]) -> list[str]:
    """Extract English (or language-unspecified) product titles."""
    titles = item.get("item_name") or []
    if isinstance(titles, str):
        return [titles]

    result: list[str] = []
    if isinstance(titles, list):
        for title in titles:
            if not isinstance(title, dict):
                continue
            language = str(title.get("language_tag", ""))
            value = title.get("value")
            if value and (language.startswith("en") or not language):
                result.append(str(value))
    return result


def get_bytes(session: requests.Session, url: str, retries: int = 4) -> bytes:
    """Download a small/medium file into memory with retry handling."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=120)
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def find_matching_listings(
    session: requests.Session,
    product_type: str,
    title_contains: str | None,
) -> list[dict[str, Any]]:
    """Return unique spin-bearing listings matching the requested category."""
    target_type = product_type.casefold()
    target_title = title_contains.casefold() if title_contains else None
    matches: list[dict[str, Any]] = []

    for shard_index, shard in enumerate(LISTING_SHARDS, start=1):
        filename = f"listings_{shard}.json.gz"
        url = f"{BASE_URL}/listings/metadata/{filename}"
        print(
            f"Reading listing metadata {shard_index}/{len(LISTING_SHARDS)} "
            f"({filename})...",
            file=sys.stderr,
        )
        compressed = get_bytes(session, url)
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as gz:
            for raw_line in gz:
                item = json.loads(raw_line)
                spin_id = item.get("spin_id")
                if not spin_id:
                    continue

                product_types = values_from_field(item.get("product_type"))
                if not any(value.casefold() == target_type for value in product_types):
                    continue

                titles = english_titles(item)
                if target_title and not any(
                    target_title in title.casefold() for title in titles
                ):
                    continue

                matches.append(
                    {
                        "item_id": item.get("item_id"),
                        "domain_name": item.get("domain_name"),
                        "spin_id": str(spin_id),
                        "product_type": product_types,
                        "titles": titles,
                    }
                )

    # A spin may theoretically be referenced by more than one listing.
    unique_by_spin: dict[str, dict[str, Any]] = {}
    for listing in matches:
        unique_by_spin.setdefault(listing["spin_id"], listing)
    return list(unique_by_spin.values())


def iter_matching_spin_rows(
    session: requests.Session, spin_ids: set[str]
) -> Iterable[dict[str, str]]:
    """Yield spin metadata rows whose spin_id is in spin_ids."""
    url = f"{BASE_URL}/spins/metadata/spins.csv.gz"
    print("Reading 360-degree image metadata...", file=sys.stderr)
    compressed = get_bytes(session, url)
    with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as gz:
        text = io.TextIOWrapper(gz, encoding="utf-8", newline="")
        for row in csv.DictReader(text):
            if row.get("spin_id") in spin_ids:
                yield row


def azimuth_key(row: dict[str, str]) -> tuple[float, str]:
    """Sort spin rows numerically by azimuth, with a stable path fallback."""
    try:
        azimuth = float(row.get("azimuth", "0"))
    except (TypeError, ValueError):
        azimuth = 0.0
    return azimuth, row.get("path", "")


def circular_distance(index_a: int, index_b: int, total: int) -> int:
    """Return shortest index distance around a circular spin sequence."""
    direct = abs(index_a - index_b)
    return min(direct, total - direct)


def evenly_spaced_indices(total_frames: int, count: int) -> list[int]:
    """Return count indices spread across the whole sequence, including ends."""
    if count < 1:
        raise ValueError("count must be at least 1")
    if count > total_frames:
        raise ValueError("count cannot exceed total_frames")
    if count == 1:
        return [0]

    last_index = total_frames - 1
    indices = [math.floor(index * last_index / (count - 1)) for index in range(count)]

    # Rounding can collide for tiny sequences. Fill any gap deterministically.
    unique_indices = []
    seen = set()
    for index in indices:
        if index not in seen:
            unique_indices.append(index)
            seen.add(index)
    for index in range(total_frames):
        if len(unique_indices) >= count:
            break
        if index not in seen:
            unique_indices.append(index)
            seen.add(index)

    return sorted(unique_indices)


def select_frame_indices(
    total_frames: int,
    requested_step: int = 6,
    min_gallery_images: int = 8,
    max_gallery_images: int | None = None,
    rng: random.Random | None = None,
) -> tuple[list[int], int, str]:
    """Select gallery indices and one held-out query index.

    For a long spin, requested_step is used directly. For a shorter spin, the
    effective step is reduced automatically so that at least
    min_gallery_images remain after one query frame is withheld.
    If max_gallery_images is set, it selects that many views evenly across the
    whole spin instead, for example 72 frames with max=4 gives frames
    1, 24, 48, 72 in 1-based numbering.

    Returns:
        (gallery_indices, query_index, sampling_summary)
    """
    if requested_step < 1:
        raise ValueError("requested_step must be at least 1")
    if min_gallery_images < 1:
        raise ValueError("min_gallery_images must be at least 1")
    if max_gallery_images is not None and max_gallery_images < 1:
        raise ValueError("max_gallery_images must be at least 1")

    required_gallery_images = max_gallery_images or min_gallery_images
    if max_gallery_images is None and total_frames < min_gallery_images + 1:
        raise ValueError(
            f"Need at least {min_gallery_images + 1} source frames to create "
            f"{min_gallery_images} gallery images plus one distinct query; "
            f"found {total_frames}."
        )

    if total_frames < required_gallery_images + 1:
        raise ValueError(
            f"Need at least {required_gallery_images + 1} source frames to create "
            f"{required_gallery_images} gallery images plus one distinct query; "
            f"found {total_frames}."
        )

    if max_gallery_images is not None:
        gallery_set = set(evenly_spaced_indices(total_frames, max_gallery_images))
        sampling_summary = f"even={max_gallery_images}"
    else:
        # Largest safe stride that still leaves enough gallery views. Capping it
        # at requested_step means 70+ frame spins use every 6th image by default.
        largest_safe_step = max(1, (total_frames - 1) // min_gallery_images)
        effective_step = min(requested_step, largest_safe_step)
        gallery_set = set(range(0, total_frames, effective_step))
        sampling_summary = f"step={effective_step}"

    # Select the query uniformly at random from frames that are not in the
    # gallery. This prevents every object from receiving the same fixed view
    # (for example, the back view) while keeping query and gallery distinct.
    if rng is None:
        rng = random.Random()

    held_out_candidates = [
        index for index in range(total_frames) if index not in gallery_set
    ]
    if held_out_candidates:
        query_index = rng.choice(held_out_candidates)
    else:
        # effective_step == 1 selected every source frame. Randomly withhold
        # one source frame so the query is still distinct from the gallery.
        query_index = rng.randrange(total_frames)
        gallery_set.remove(query_index)

    # Defensive backfill for unusual edge cases. Normally the adaptive stride
    # already guarantees the requested minimum.
    if max_gallery_images is None and len(gallery_set) < min_gallery_images:
        for index in range(total_frames):
            if index != query_index:
                gallery_set.add(index)
            if len(gallery_set) >= min_gallery_images:
                break

    gallery_indices = sorted(gallery_set)
    if query_index in gallery_set:
        raise AssertionError("Query frame must not also appear in the gallery")
    if max_gallery_images is None and len(gallery_indices) < min_gallery_images:
        raise AssertionError("Sampling failed to produce enough gallery images")
    if max_gallery_images is not None and len(gallery_indices) != max_gallery_images:
        raise AssertionError("Sampling failed to produce the requested gallery image count")

    return gallery_indices, query_index, sampling_summary


def build_object_plans(
    listings: list[dict[str, Any]],
    rows_by_spin: dict[str, list[dict[str, str]]],
    requested_step: int,
    min_gallery_images: int,
    max_gallery_images: int | None,
    max_objects: int | None,
    rng: random.Random,
) -> list[ObjectPlan]:
    """Build valid, sequential object plans, skipping undersized spins."""
    plans: list[ObjectPlan] = []

    for listing in listings:
        spin_id = listing["spin_id"]
        rows = sorted(rows_by_spin.get(spin_id, []), key=azimuth_key)
        try:
            gallery_indices, query_index, sampling_summary = select_frame_indices(
                total_frames=len(rows),
                requested_step=requested_step,
                min_gallery_images=min_gallery_images,
                max_gallery_images=max_gallery_images,
                rng=rng,
            )
        except ValueError as exc:
            print(f"Skipping spin {spin_id}: {exc}", file=sys.stderr)
            continue

        plans.append(
            ObjectPlan(
                listing=listing,
                source_count=len(rows),
                sampling_summary=sampling_summary,
                gallery_rows=[rows[index] for index in gallery_indices],
                query_row=rows[query_index],
            )
        )
        if max_objects is not None and len(plans) >= max_objects:
            break

    return plans


def download_file(
    session: requests.Session,
    url: str,
    destination: Path,
    retries: int = 4,
) -> None:
    """Stream one image to disk atomically, skipping valid existing files."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return

    last_error: Exception | None = None
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(retries):
        try:
            with session.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with temporary.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file.write(chunk)
                temporary.replace(destination)
            return
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(2**attempt)

    raise RuntimeError(f"Could not download {url}: {last_error}")


def row_url(row: dict[str, str]) -> str:
    """Return the public S3 URL for one original ABO spin image."""
    path = row.get("path")
    if not path:
        raise ValueError(f"Spin metadata row has no image path: {row}")
    return f"{BASE_URL}/spins/original/{path}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download selected ABO spins into gallery/object_NNN and "
            "query/query_NNN.jpg retrieval-dataset folders."
        )
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {SCRIPT_VERSION}",
    )
    parser.add_argument(
        "--product-type",
        default="SOFA",
        help="Exact ABO product_type value (default: SOFA).",
    )
    parser.add_argument(
        "--title-contains",
        help="Optional case-insensitive phrase required in an English title.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workshop_materials/data"),
        help="Dataset root containing gallery/ and query/.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        help="Create at most this many valid objects, useful for testing.",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=6,
        help=(
            "Take every Nth frame for long spins (default: 6). The step is "
            "automatically reduced for short spins to preserve the minimum."
        ),
    )
    parser.add_argument(
        "--min-gallery-images",
        type=int,
        default=8,
        help="Minimum images in every object gallery folder (default: 8).",
    )
    parser.add_argument(
        "--max-gallery-images",
        type=int,
        help=(
            "Maximum gallery images per object. When set, frames are sampled "
            "evenly across the full spin; for example 72 frames with 4 gives "
            "1, 24, 48, 72 in 1-based numbering."
        ),
    )
    parser.add_argument(
        "--write-query-labels",
        action="store_true",
        help="Write OUTPUT/query_labels.csv with query-to-object ground truth.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help=(
            "Optional random seed for reproducible query-image selection. "
            "Without it, query views are randomized on every run."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the planned sampling without downloading any images.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_objects is not None and args.max_objects < 1:
        raise ValueError("--max-objects must be at least 1")
    if args.sample_step < 1:
        raise ValueError("--sample-step must be at least 1")
    if args.min_gallery_images < 1:
        raise ValueError("--min-gallery-images must be at least 1")
    if args.max_gallery_images is not None and args.max_gallery_images < 1:
        raise ValueError("--max-gallery-images must be at least 1")


def main() -> int:
    args = parse_args()
    validate_args(args)

    with requests.Session() as session:
        session.headers.update({"User-Agent": "ABO-retrieval-dataset-builder/3.0"})

        listings = find_matching_listings(
            session=session,
            product_type=args.product_type,
            title_contains=args.title_contains,
        )
        if not listings:
            print(
                f"No products found for product_type={args.product_type!r}",
                file=sys.stderr,
            )
            return 2

        spin_ids = {listing["spin_id"] for listing in listings}
        rows_by_spin: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in iter_matching_spin_rows(session, spin_ids):
            rows_by_spin[row["spin_id"]].append(row)

        rng = random.Random(args.seed)
        plans = build_object_plans(
            listings=listings,
            rows_by_spin=rows_by_spin,
            requested_step=args.sample_step,
            min_gallery_images=args.min_gallery_images,
            max_gallery_images=args.max_gallery_images,
            max_objects=args.max_objects,
            rng=rng,
        )
        if not plans:
            print(
                "No matching spins contained enough frames for the requested "
                "gallery/query split.",
                file=sys.stderr,
            )
            return 3

        total_gallery = sum(len(plan.gallery_rows) for plan in plans)
        print(
            f"Planned {len(plans)} objects, {total_gallery} gallery images, "
            f"and {len(plans)} query images.",
            file=sys.stderr,
        )
        for object_number, plan in enumerate(plans, start=1):
            print(
                f"  object_{object_number:03d}: {plan.source_count} source frames, "
                f"{plan.sampling_summary}, "
                f"gallery={len(plan.gallery_rows)}, query=1",
                file=sys.stderr,
            )

        if args.dry_run:
            return 0

        gallery_root = args.output / "gallery"
        query_root = args.output / "query"
        gallery_root.mkdir(parents=True, exist_ok=True)
        query_root.mkdir(parents=True, exist_ok=True)

        labels: list[dict[str, str]] = []
        total_downloads = total_gallery + len(plans)
        completed_downloads = 0

        for object_number, plan in enumerate(plans, start=1):
            object_name = f"object_{object_number:03d}"
            query_name = f"query_{object_number:03d}.jpg"
            object_folder = gallery_root / object_name
            query_path = query_root / query_name

            # Rebuild this numbered object cleanly so stale files from an older
            # run cannot inflate its image count or mismatch its query.
            shutil.rmtree(object_folder, ignore_errors=True)
            query_path.unlink(missing_ok=True)
            object_folder.mkdir(parents=True, exist_ok=True)

            try:
                for image_number, row in enumerate(plan.gallery_rows, start=1):
                    destination = object_folder / f"img_{image_number:03d}.jpg"
                    completed_downloads += 1
                    print(
                        f"[{completed_downloads}/{total_downloads}] {destination}",
                        file=sys.stderr,
                    )
                    download_file(session, row_url(row), destination)

                completed_downloads += 1
                print(
                    f"[{completed_downloads}/{total_downloads}] {query_path}",
                    file=sys.stderr,
                )
                download_file(session, row_url(plan.query_row), query_path)
            except Exception:
                # Do not leave an object that violates the minimum-image rule.
                shutil.rmtree(object_folder, ignore_errors=True)
                query_path.unlink(missing_ok=True)
                raise

            gallery_count = sum(
                1
                for path in object_folder.glob("img_*.jpg")
                if path.is_file() and path.stat().st_size > 0
            )
            required_gallery_count = args.max_gallery_images or args.min_gallery_images
            if gallery_count < required_gallery_count:
                shutil.rmtree(object_folder, ignore_errors=True)
                query_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"{object_name} ended with only {gallery_count} gallery "
                    f"images; required at least {required_gallery_count}."
                )

            labels.append({"query_name": query_name, "true_object": object_name})

        labels_path = args.output / "query_labels.csv"
        if args.write_query_labels:
            with labels_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file, fieldnames=["query_name", "true_object"]
                )
                writer.writeheader()
                writer.writerows(labels)
            print(f"Wrote labels: {labels_path}", file=sys.stderr)
        else:
            # Keep the flag meaningful on reruns: without it, do not leave a
            # stale labels file that may no longer match the generated dataset.
            labels_path.unlink(missing_ok=True)

    print(f"Done. Dataset saved under: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
