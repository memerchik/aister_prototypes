#!/usr/bin/env python3
"""Create single-item Top-3 retrieval visualizations from saved results.

The script reads the cached query result CSVs, local query images, and local
gallery images. By default it embeds only the selected query images so it can
recover the exact gallery image that produced each Top-k object match.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import math
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO_DIR = Path(__file__).resolve().parents[2]
STEP_DIR = REPO_DIR / "step_1"
DATASET_DIR = STEP_DIR / "data" / "dataset_dev"
QUERY_DIR = DATASET_DIR / "query"
GALLERY_DIR = DATASET_DIR / "gallery"
OUTPUTS_DIR = STEP_DIR / "outputs"
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "single_item_visualizations"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MODEL_NAME = "facebook/dinov2-base"
BACKGROUND_REMOVAL_MODEL = "u2net"
BACKGROUND_FILL_COLOR = (255, 255, 255)
DEFAULT_TOP_K_IMAGES = 30

VARIANT_SPECS = {
    "raw": {
        "label": "Raw baseline",
        "results": OUTPUTS_DIR / "query_results_fixed.csv",
        "metadata": OUTPUTS_DIR / "gallery_index_metadata.csv",
        "embeddings": OUTPUTS_DIR / "gallery_embeddings.npy",
        "background_removed": False,
    },
    "bg": {
        "label": "Background removed",
        "results": OUTPUTS_DIR / "query_results_bg_removed.csv",
        "metadata": OUTPUTS_DIR / "gallery_index_metadata_bg_removed.csv",
        "embeddings": OUTPUTS_DIR / "gallery_embeddings_bg_removed.npy",
        "background_removed": True,
    },
}

CURATED_EXAMPLES = {
    "raw": [
        ("clean_success", "img_030.JPG"),
        ("top1_correct_but_rejected", "img_027.JPG"),
        ("wrong_top1_true_in_top3", "img_077.png"),
        ("unknown_correctly_rejected", "img_067.JPG"),
        ("unknown_false_approval", "img_064.JPG"),
    ],
    "bg": [
        ("bg_success", "img_046.JPG"),
        ("bg_success_from_raw_failure", "img_019.JPG"),
        ("bg_can_hurt", "img_035.JPG"),
        ("bg_top3_but_rejected", "img_027.JPG"),
        ("bg_unknown_false_approval", "img_066.JPG"),
    ],
}

CATEGORY_ORDER = [
    "correct_approval",
    "truth_in_top3_not_approved",
    "known_rejection",
    "wrong_approval",
    "correct_unknown_rejection",
    "false_unknown_approval",
]

COLORS = {
    "paper": "#FFFFFF",
    "card": "#FFFFFF",
    "panel": "#FFF8EB",
    "picture_bg": "#FFFFFF",
    "ink": "#071951",
    "muted": "#56607A",
    "line": "#CDD5E8",
    "success": "#1C8B5A",
    "warning": "#F9B22F",
    "danger": "#FF5757",
    "blue": "#071951",
}


@dataclass(frozen=True)
class MatchEvidence:
    object_id: str
    path: Path
    image_name: str
    image_relpath: str
    score: float


def is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value)


def clean_object_id(value: object) -> str | None:
    if is_missing(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def parse_ranked_ids(value: object) -> list[str]:
    if is_missing(value):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def parse_float_list(value: object) -> list[float]:
    if is_missing(value):
        return []
    if isinstance(value, list):
        raw_values = value
    else:
        try:
            raw_values = ast.literal_eval(str(value))
        except (SyntaxError, ValueError):
            return []
    if not isinstance(raw_values, list):
        return []

    result = []
    for item in raw_values:
        if is_missing(item):
            result.append(np.nan)
            continue
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            result.append(np.nan)
    return result


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def ensure_metadata_paths(metadata_path: Path) -> pd.DataFrame:
    metadata_df = pd.read_csv(metadata_path)
    if "image_relpath" not in metadata_df.columns:
        if {"object_id", "image_name"}.issubset(metadata_df.columns):
            metadata_df["image_relpath"] = metadata_df.apply(
                lambda row: str(Path(str(row["object_id"])) / str(row["image_name"])),
                axis=1,
            )
        else:
            raise ValueError(f"{metadata_path} must contain image_relpath or object_id/image_name columns.")
    if "image_name" not in metadata_df.columns:
        metadata_df["image_name"] = metadata_df["image_relpath"].apply(lambda p: Path(str(p)).name)
    return metadata_df


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "DejaVuSans-Bold.ttf",
        ]
        if bold
        else [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "DejaVuSans.ttf",
        ]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONTS = {
    "title": load_font(44, bold=True),
    "subtitle": load_font(25),
    "section": load_font(28, bold=True),
    "body": load_font(24),
    "small": load_font(18),
    "tiny": load_font(15),
    "badge": load_font(18, bold=True),
}


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            if text_size(draw, candidate, font)[0] <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    line_gap: int = 6,
) -> int:
    x, y = xy
    for line in wrap_text(draw, text, font, max_width):
        draw.text((x, y), line, font=font, fill=fill)
        y += text_size(draw, line, font)[1] + line_gap
    return y


def draw_pill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    fill: str,
    outline: str | None = None,
    text_fill: str = "white",
    font: ImageFont.ImageFont = FONTS["badge"],
    pad_x: int = 16,
    pad_y: int = 8,
) -> tuple[int, int, int, int]:
    x, y = xy
    width, height = text_size(draw, text, font)
    box = (x, y, x + width + pad_x * 2, y + height + pad_y * 2)
    draw.rounded_rectangle(box, radius=14, fill=fill, outline=outline, width=2)
    draw.text((x + pad_x, y + pad_y - 1), text, font=font, fill=text_fill)
    return box


def pill_text_color(fill: str) -> str:
    return COLORS["ink"] if fill in {COLORS["warning"], COLORS["danger"]} else "#FFFFFF"


def open_image(path: Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    if image.mode not in {"RGB", "RGBA"}:
        image = image.convert("RGB")
    return image


def image_on_background(path: Path | None, size: tuple[int, int], bg: str = COLORS["picture_bg"]) -> Image.Image:
    canvas = Image.new("RGB", size, bg)
    if path is None or not path.exists():
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=COLORS["line"], width=3)
        message = "image not found"
        w, h = text_size(draw, message, FONTS["body"])
        draw.text(((size[0] - w) // 2, (size[1] - h) // 2), message, font=FONTS["body"], fill=COLORS["muted"])
        return canvas

    image = open_image(path).convert("RGBA")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y), image)
    return canvas


def montage(paths: list[Path], size: tuple[int, int], bg: str = COLORS["picture_bg"]) -> Image.Image:
    if len(paths) <= 1:
        return image_on_background(paths[0] if paths else None, size, bg=bg)

    count = min(len(paths), 4)
    columns = 2 if count > 1 else 1
    rows = math.ceil(count / columns)
    gap = 8
    cell_w = (size[0] - gap * (columns - 1)) // columns
    cell_h = (size[1] - gap * (rows - 1)) // rows
    canvas = Image.new("RGB", size, bg)
    for index, path in enumerate(paths[:count]):
        row, col = divmod(index, columns)
        thumb = image_on_background(path, (cell_w, cell_h), bg=bg)
        canvas.paste(thumb, (col * (cell_w + gap), row * (cell_h + gap)))
    return canvas


def build_gallery_lookup(metadata_path: Path, gallery_dir: Path) -> dict[str, list[Path]]:
    lookup: dict[str, list[Path]] = {}
    if metadata_path.exists():
        metadata_df = ensure_metadata_paths(metadata_path)
        for object_id, rows in metadata_df.groupby("object_id", sort=False):
            paths = []
            for relpath in rows["image_relpath"].dropna():
                path = gallery_dir / str(relpath)
                if path.exists():
                    paths.append(path)
            if paths:
                lookup[str(object_id)] = paths

    if gallery_dir.exists():
        for folder in sorted(gallery_dir.iterdir()):
            if not folder.is_dir():
                continue
            paths = sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
            lookup.setdefault(folder.name, paths)

    return lookup


class ExactMatchResolver:
    """Recover exact gallery evidence for candidate objects from cached embeddings."""

    def __init__(self, *, top_k_images: int):
        self.top_k_images = top_k_images
        self.processor = None
        self.model = None
        self.torch = None
        self.device = None
        self.remove_background = None
        self.background_session = None
        self.variant_cache: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}

    def load_model(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "Exact matched-image visualizations need `torch` and `transformers`. "
                "Run the script from the same environment as the notebooks, or use --match-source folder-first."
            ) from exc

        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self.torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(device)
        self.model.eval()

    def get_background_removal(self):
        if self.remove_background is not None:
            return self.remove_background, self.background_session
        try:
            from rembg import new_session, remove
        except ImportError as exc:
            raise ImportError(
                "Background-removed exact matched-image visualizations need `rembg` and `onnxruntime`. "
                "Run the script from the background-removal notebook environment, or use --variant raw."
            ) from exc
        self.remove_background = remove
        self.background_session = new_session(BACKGROUND_REMOVAL_MODEL)
        return self.remove_background, self.background_session

    def preprocess(self, image: Image.Image, *, background_removed: bool) -> Image.Image:
        image = image.convert("RGB")
        if not background_removed:
            return image

        remove, session = self.get_background_removal()
        foreground = remove(image.convert("RGBA"), session=session)
        if isinstance(foreground, Image.Image):
            foreground = foreground.convert("RGBA")
        elif isinstance(foreground, bytes):
            foreground = Image.open(io.BytesIO(foreground)).convert("RGBA")
        else:
            foreground = Image.fromarray(foreground).convert("RGBA")

        canvas = Image.new("RGBA", foreground.size, BACKGROUND_FILL_COLOR + (255,))
        canvas.alpha_composite(foreground)
        return canvas.convert("RGB")

    def embed_query(self, query_path: Path, *, background_removed: bool) -> np.ndarray:
        self.load_model()
        assert self.processor is not None
        assert self.model is not None
        assert self.torch is not None
        assert self.device is not None

        image = self.preprocess(open_image(query_path), background_removed=background_removed)
        inputs = self.processor(images=[image], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self.torch.inference_mode():
            outputs = self.model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embedding = outputs.pooler_output
            else:
                embedding = outputs.last_hidden_state.mean(dim=1)

        embedding_np = embedding.detach().to("cpu").float().numpy().astype("float32")
        return l2_normalize(embedding_np)[0]

    def load_variant(self, variant: str) -> tuple[pd.DataFrame, np.ndarray]:
        if variant in self.variant_cache:
            return self.variant_cache[variant]

        spec = VARIANT_SPECS[variant]
        metadata_path = spec["metadata"]
        embeddings_path = spec["embeddings"]
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing gallery metadata: {metadata_path}")
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Missing gallery embeddings: {embeddings_path}")

        metadata_df = ensure_metadata_paths(metadata_path)
        embeddings = np.load(embeddings_path).astype("float32")
        embeddings = l2_normalize(embeddings)
        if len(metadata_df) != len(embeddings):
            raise ValueError(
                f"Metadata/embedding length mismatch for {variant}: "
                f"{len(metadata_df)} metadata rows vs {len(embeddings)} embeddings."
            )

        self.variant_cache[variant] = (metadata_df, embeddings)
        return metadata_df, embeddings

    def resolve(self, *, row: pd.Series, variant: str) -> dict[str, MatchEvidence]:
        query_path = QUERY_DIR / str(row["query_name"])
        metadata_df, embeddings = self.load_variant(variant)
        query_embedding = self.embed_query(
            query_path,
            background_removed=bool(VARIANT_SPECS[variant]["background_removed"]),
        )

        scores = embeddings @ query_embedding
        top_count = min(self.top_k_images, len(scores))
        if top_count <= 0:
            return {}

        candidate_indices = np.argpartition(-scores, top_count - 1)[:top_count]
        candidate_indices = candidate_indices[np.argsort(-scores[candidate_indices])]

        best_by_object: dict[str, MatchEvidence] = {}
        for idx in candidate_indices:
            rec = metadata_df.iloc[int(idx)]
            object_id = str(rec["object_id"])
            if object_id in best_by_object:
                continue

            image_relpath = str(rec["image_relpath"])
            image_path = GALLERY_DIR / image_relpath
            best_by_object[object_id] = MatchEvidence(
                object_id=object_id,
                path=image_path,
                image_name=str(rec["image_name"]),
                image_relpath=image_relpath,
                score=float(scores[idx]),
            )

        return best_by_object


def classify_row(row: pd.Series, ranked_ids: list[str]) -> tuple[str, str, str]:
    is_known = bool(row["is_known"])
    true_object = clean_object_id(row.get("true_object"))
    pred_object = clean_object_id(row.get("pred_object"))

    if is_known and pred_object == true_object:
        return "correct_approval", "Correct approval", COLORS["success"]
    if not is_known and pred_object is None:
        return "correct_unknown_rejection", "Correct unknown rejection", COLORS["success"]
    if is_known and true_object in ranked_ids[:3]:
        return "truth_in_top3_not_approved", "Truth in Top-3, final missed", COLORS["warning"]
    if is_known and pred_object is None:
        return "known_rejection", "Known object rejected", COLORS["warning"]
    if not is_known and pred_object is not None:
        return "false_unknown_approval", "False unknown approval", COLORS["danger"]
    return "wrong_approval", "Wrong approval", COLORS["danger"]


def format_score(value: object, digits: int = 3) -> str:
    if is_missing(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def draw_metric_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    value: str,
    accent: str,
) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=18, fill=COLORS["panel"], outline=COLORS["line"], width=2)
    draw.rectangle((x1, y1, x1 + 8, y2), fill=accent)
    draw.text((x1 + 24, y1 + 17), label.upper(), font=FONTS["tiny"], fill=COLORS["muted"])
    draw.text((x1 + 24, y1 + 45), value, font=FONTS["section"], fill=COLORS["ink"])


def draw_candidate_card(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    *,
    rank: int,
    object_id: str,
    image_path: Path | None,
    image_name: str | None,
    match_score: float | None,
    source_label: str,
    true_object: str | None,
    pred_object: str | None,
    is_known: bool,
    score_text: str | None,
) -> None:
    draw = ImageDraw.Draw(canvas)
    x1, y1, x2, y2 = box
    is_true = is_known and object_id == true_object
    is_pred = pred_object is not None and object_id == pred_object
    if is_true and is_pred:
        outline = COLORS["success"]
    elif is_true:
        outline = COLORS["success"]
    elif is_pred:
        outline = COLORS["danger"] if not is_true else COLORS["success"]
    else:
        outline = COLORS["line"]

    draw.rounded_rectangle(box, radius=28, fill=COLORS["card"], outline=outline, width=5 if outline != COLORS["line"] else 2)
    draw.text((x1 + 24, y1 + 22), f"Top {rank}", font=FONTS["section"], fill=COLORS["ink"])
    draw.text((x1 + 24, y1 + 62), object_id, font=FONTS["body"], fill=COLORS["muted"])
    if image_name:
        draw.text((x1 + 24, y1 + 91), f"{source_label}: {image_name}", font=FONTS["tiny"], fill=COLORS["muted"])

    pill_x = x1 + 24
    pill_y = y1 + 116
    if is_true:
        pill = draw_pill(draw, (pill_x, pill_y), "TRUE OBJECT", fill=COLORS["success"])
        pill_x = pill[2] + 10
    if is_pred:
        draw_pill(draw, (pill_x, pill_y), "FINAL PICK", fill=COLORS["blue"])

    if score_text:
        score_w, _ = text_size(draw, score_text, FONTS["small"])
        draw.text((x2 - score_w - 24, y1 + 34), score_text, font=FONTS["small"], fill=COLORS["muted"])
    if match_score is not None:
        match_text = f"image score {match_score:.3f}"
        match_w, _ = text_size(draw, match_text, FONTS["tiny"])
        draw.text((x2 - match_w - 24, y1 + 64), match_text, font=FONTS["tiny"], fill=COLORS["muted"])

    image_box = (x1 + 24, y1 + 168, x2 - 24, y2 - 34)
    image = image_on_background(image_path, (image_box[2] - image_box[0], image_box[3] - image_box[1]))
    canvas.paste(image, (image_box[0], image_box[1]))
    draw.rounded_rectangle(image_box, radius=18, outline=COLORS["line"], width=2)


def draw_truth_panel(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    *,
    true_object: str | None,
    truth_evidence: MatchEvidence | None,
    fallback_path: Path | None,
) -> str:
    draw = ImageDraw.Draw(canvas)
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=18, fill=COLORS["panel"], outline=COLORS["line"], width=2)

    if true_object is None:
        draw.text((x1 + 18, y1 + 18), "Ground truth", font=FONTS["small"], fill=COLORS["ink"])
        draw_wrapped_text(
            draw,
            (x1 + 18, y1 + 52),
            "Unknown query: there is no correct gallery object.",
            FONTS["small"],
            COLORS["muted"],
            x2 - x1 - 36,
        )
        return ""

    truth_path = truth_evidence.path if truth_evidence else fallback_path
    truth_name = truth_evidence.image_name if truth_evidence else (fallback_path.name if fallback_path else None)
    truth_source = "exact retrieved image" if truth_evidence else "reference image"
    image_box = (x1 + 18, y1 + 52, x1 + 178, y2 - 18)
    image = image_on_background(truth_path, (image_box[2] - image_box[0], image_box[3] - image_box[1]))
    canvas.paste(image, (image_box[0], image_box[1]))
    draw.rounded_rectangle(image_box, radius=12, outline=COLORS["line"], width=2)
    draw.text((x1 + 18, y1 + 18), "Correct object", font=FONTS["small"], fill=COLORS["ink"])
    draw.text((x1 + 198, y1 + 54), true_object, font=FONTS["body"], fill=COLORS["success"])
    if truth_name:
        draw_wrapped_text(draw, (x1 + 198, y1 + 91), f"{truth_source}: {truth_name}", FONTS["tiny"], COLORS["muted"], x2 - x1 - 220)
    return str(truth_path) if truth_path else ""


def make_card(
    *,
    row: pd.Series,
    variant: str,
    category_hint: str,
    gallery_lookup: dict[str, list[Path]],
    exact_matches: dict[str, MatchEvidence],
    output_dir: Path,
    top_k: int,
    score_threshold: float,
    margin_threshold: float,
) -> dict[str, str]:
    ranked_ids = parse_ranked_ids(row["ranked_ids"])[:top_k]
    ranked_scores = parse_float_list(row.get("top3_scores"))[:top_k]
    category, outcome, outcome_color = classify_row(row, ranked_ids)
    category = category_hint or category

    query_name = str(row["query_name"])
    query_path = QUERY_DIR / query_name
    true_object = clean_object_id(row.get("true_object"))
    pred_object = clean_object_id(row.get("pred_object"))
    is_known = bool(row["is_known"])
    variant_label = VARIANT_SPECS[variant]["label"]
    truth_label = true_object if is_known and true_object else "unknown"
    pred_label = pred_object if pred_object else "rejected"

    canvas = Image.new("RGB", (1920, 1080), COLORS["paper"])
    draw = ImageDraw.Draw(canvas)

    title = f"{variant_label} | {query_name}"
    draw.text((60, 44), title, font=FONTS["title"], fill=COLORS["ink"])
    draw_wrapped_text(
        draw,
        (62, 100),
        f"Truth: {truth_label}   |   Model decision: {pred_label}",
        FONTS["subtitle"],
        COLORS["muted"],
        1200,
    )
    draw_pill(draw, (1460, 54), outcome, fill=outcome_color, text_fill=pill_text_color(outcome_color))

    query_box = (60, 170, 540, 830)
    draw.rounded_rectangle(query_box, radius=28, fill=COLORS["card"], outline=COLORS["line"], width=2)
    draw.text((90, 200), "Query image", font=FONTS["section"], fill=COLORS["ink"])
    query_image_box = (90, 255, 510, 590)
    query_image = image_on_background(query_path, (query_image_box[2] - query_image_box[0], query_image_box[3] - query_image_box[1]))
    canvas.paste(query_image, (query_image_box[0], query_image_box[1]))
    draw.rounded_rectangle(query_image_box, radius=18, outline=COLORS["line"], width=2)

    truth_fallback = gallery_lookup.get(true_object or "", [None])[0] if true_object else None
    truth_evidence = exact_matches.get(true_object) if true_object else None
    truth_image_path = draw_truth_panel(
        canvas,
        (90, 625, 510, 805),
        true_object=true_object if is_known else None,
        truth_evidence=truth_evidence,
        fallback_path=truth_fallback,
    )

    candidate_x = 600
    candidate_y = 170
    candidate_w = 405
    candidate_h = 660
    gap = 30
    for index, object_id in enumerate(ranked_ids):
        x1 = candidate_x + index * (candidate_w + gap)
        box = (x1, candidate_y, x1 + candidate_w, candidate_y + candidate_h)
        evidence = exact_matches.get(object_id)
        candidate_score = ranked_scores[index] if index < len(ranked_scores) else None
        if is_missing(candidate_score):
            if index == 0:
                candidate_score = row.get("best_score")
            elif index == 1:
                candidate_score = row.get("second_score")
        if is_missing(candidate_score) and evidence is not None:
            candidate_score = evidence.score
        score_text = f"score {format_score(candidate_score)}"
        fallback = gallery_lookup.get(object_id, [None])[0]
        image_path = evidence.path if evidence else fallback
        image_name = evidence.image_name if evidence else (fallback.name if fallback else None)
        source_label = "matched image" if evidence else "reference image"
        draw_candidate_card(
            canvas,
            box,
            rank=index + 1,
            object_id=object_id,
            image_path=image_path,
            image_name=image_name,
            match_score=evidence.score if evidence else None,
            source_label=source_label,
            true_object=true_object,
            pred_object=pred_object,
            is_known=is_known,
            score_text=score_text,
        )

    if len(ranked_ids) < top_k:
        for index in range(len(ranked_ids), top_k):
            x1 = candidate_x + index * (candidate_w + gap)
            box = (x1, candidate_y, x1 + candidate_w, candidate_y + candidate_h)
            draw.rounded_rectangle(box, radius=28, fill=COLORS["card"], outline=COLORS["line"], width=2)
            draw.text((x1 + 24, candidate_y + 22), f"Top {index + 1}", font=FONTS["section"], fill=COLORS["muted"])
            draw.text((x1 + 24, candidate_y + 70), "not available", font=FONTS["body"], fill=COLORS["muted"])

    metric_y = 870
    top1_score = ranked_scores[0] if len(ranked_scores) > 0 else row.get("best_score")
    top2_score = ranked_scores[1] if len(ranked_scores) > 1 else row.get("second_score")
    draw_metric_box(draw, (60, metric_y, 380, 1015), "Top-1 score", format_score(top1_score), COLORS["blue"])
    draw_metric_box(draw, (410, metric_y, 730, 1015), "Top-2 score", format_score(top2_score), COLORS["blue"])
    draw_metric_box(draw, (760, metric_y, 1080, 1015), "Margin", format_score(row.get("margin"), 4), outcome_color)
    rule_box = (1110, metric_y, 1495, 1015)
    draw.rounded_rectangle(rule_box, radius=18, fill=COLORS["panel"], outline=COLORS["line"], width=2)
    draw.rectangle((rule_box[0], rule_box[1], rule_box[0] + 8, rule_box[3]), fill=COLORS["warning"])
    draw.text((rule_box[0] + 24, rule_box[1] + 17), "DISPLAYED RULE", font=FONTS["tiny"], fill=COLORS["muted"])
    draw.text((rule_box[0] + 24, rule_box[1] + 45), f"score >= {score_threshold:.2f}", font=FONTS["body"], fill=COLORS["ink"])
    draw.text((rule_box[0] + 24, rule_box[1] + 82), f"margin >= {margin_threshold:.3f}", font=FONTS["body"], fill=COLORS["ink"])
    draw.rounded_rectangle((1525, metric_y, 1860, 1015), radius=18, fill=COLORS["panel"], outline=COLORS["line"], width=2)
    note = "Top-3 visualizations show the exact retrieved gallery image when cached embeddings are available."
    draw_wrapped_text(draw, (1548, metric_y + 26), note, FONTS["small"], COLORS["muted"], 285)

    safe_query_name = Path(query_name).stem
    output_name = f"{variant}_{category}_{safe_query_name}.png".replace(" ", "_")
    output_path = output_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)

    return {
        "variant": variant,
        "category": category,
        "query_name": query_name,
        "true_object": truth_label,
        "pred_object": pred_label,
        "ranked_ids": ", ".join(ranked_ids),
        "matched_image_relpaths": " | ".join(
            exact_matches[object_id].image_relpath for object_id in ranked_ids if object_id in exact_matches
        ),
        "truth_image_path": truth_image_path,
        "best_score": format_score(row.get("best_score")),
        "second_score": format_score(row.get("second_score")),
        "margin": format_score(row.get("margin"), 4),
        "outcome": outcome,
        "visualization_path": str(output_path),
    }


def rows_by_query(df: pd.DataFrame, query_names: Iterable[str]) -> list[tuple[str, pd.Series]]:
    rows = []
    indexed = df.set_index("query_name", drop=False)
    for query_name in query_names:
        if query_name not in indexed.index:
            print(f"Warning: query not found in results CSV: {query_name}", file=sys.stderr)
            continue
        row = indexed.loc[query_name]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        rows.append(("custom", row))
    return rows


def category_for_row(row: pd.Series) -> str:
    ranked_ids = parse_ranked_ids(row["ranked_ids"])
    category, _, _ = classify_row(row, ranked_ids)
    return category


def normalize_results_df(df: pd.DataFrame, *, score_threshold: float, margin_threshold: float) -> pd.DataFrame:
    """Support both full prototype results and the simpler workshop CSV."""
    df = df.copy()
    if "ranked_ids" not in df.columns and "top3_objects" in df.columns:
        df["ranked_ids"] = df["top3_objects"]
    if "best_score" not in df.columns and "top1_score" in df.columns:
        df["best_score"] = df["top1_score"]
    if "second_score" not in df.columns and "top3_scores" in df.columns:
        df["second_score"] = df["top3_scores"].apply(
            lambda value: parse_float_list(value)[1] if len(parse_float_list(value)) > 1 else np.nan
        )
    if "second_score" not in df.columns:
        df["second_score"] = np.nan
    if "margin" not in df.columns and {"best_score", "second_score"}.issubset(df.columns):
        df["margin"] = pd.to_numeric(df["best_score"], errors="coerce") - pd.to_numeric(df["second_score"], errors="coerce")
    if "margin" not in df.columns:
        df["margin"] = np.nan
    if "pred_object" not in df.columns and "top1_object" in df.columns:
        best_scores = pd.to_numeric(df["best_score"], errors="coerce")
        margins = pd.to_numeric(df["margin"], errors="coerce")
        accepted = best_scores.ge(score_threshold) & margins.ge(margin_threshold)
        df["pred_object"] = df["top1_object"].where(accepted, None)
    if "is_known" not in df.columns and "true_object" in df.columns:
        df["is_known"] = df["true_object"].apply(lambda value: clean_object_id(value) is not None)

    required = {"query_name", "true_object", "pred_object", "ranked_ids", "best_score", "second_score", "margin", "is_known"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Results CSV is missing required columns after normalization: {missing}")
    return df


def select_rows(df: pd.DataFrame, variant: str, args: argparse.Namespace) -> list[tuple[str, pd.Series]]:
    if args.queries:
        selected = rows_by_query(df, args.queries)
        return selected[: args.limit] if args.limit else selected

    if args.examples == "curated":
        selected = []
        for category, query_name in CURATED_EXAMPLES.get(variant, []):
            found = rows_by_query(df, [query_name])[:1]
            if found:
                selected.append((category, found[0][1]))
        return selected[: args.limit] if args.limit else selected

    if args.examples == "first":
        rows = [("first_rows", row) for _, row in df.head(args.limit or len(df)).iterrows()]
        return rows

    df = df.copy()
    df["_category"] = df.apply(category_for_row, axis=1)
    categories = args.categories or CATEGORY_ORDER
    selected = []
    for category in categories:
        category_df = df[df["_category"].eq(category)]
        if category_df.empty:
            continue
        if args.examples == "all":
            rows = category_df.iterrows()
        else:
            rows = category_df.head(args.max_per_category).iterrows()
        for _, row in rows:
            selected.append((category, row))
            if args.limit and len(selected) >= args.limit:
                return selected
    return selected


def write_index(output_dir: Path, rows: list[dict[str, str]]) -> Path:
    index_path = output_dir / "index.csv"
    fieldnames = [
        "variant",
        "category",
        "query_name",
        "true_object",
        "pred_object",
        "ranked_ids",
        "matched_image_relpaths",
        "truth_image_path",
        "best_score",
        "second_score",
        "margin",
        "outcome",
        "visualization_path",
    ]
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create single-item retrieval visualizations from saved Step 1 result CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python step_1/scripts/make_single_item_visualizations.py
              python step_1/scripts/make_single_item_visualizations.py --variant raw --queries img_030.JPG img_077.png
              python step_1/scripts/make_single_item_visualizations.py --examples category --categories wrong_approval false_unknown_approval
              python step_1/scripts/make_single_item_visualizations.py --variant raw --examples first --limit 20 --output-dir step_1/outputs/visualizations
              python step_1/scripts/make_single_item_visualizations.py --variant raw --examples first --limit 20 --query-dir step_1/workshop_materials/data/query --gallery-dir step_1/workshop_materials/data/gallery --results-path step_1/workshop_materials/outputs/query_results.csv --metadata-path step_1/workshop_materials/outputs/gallery_metadata.csv --embeddings-path step_1/workshop_materials/outputs/gallery_embeddings.npy --output-dir step_1/workshop_materials/outputs/visualizations
              python step_1/scripts/make_single_item_visualizations.py --variant bg --score-threshold 0.54 --margin-threshold 0.015
              python step_1/scripts/make_single_item_visualizations.py --match-source folder-first
            """
        ),
    )
    parser.add_argument("--variant", choices=["raw", "bg", "both"], default="both")
    parser.add_argument(
        "--examples",
        choices=["curated", "category", "all", "first"],
        default="curated",
        help="curated: built-in teaching examples; first: first rows in CSV order; category: first N per category; all: every result row by category.",
    )
    parser.add_argument("--queries", nargs="*", help="Specific query filenames to export, for example img_030.JPG.")
    parser.add_argument("--categories", nargs="*", choices=CATEGORY_ORDER, help="Categories used with --examples category/all.")
    parser.add_argument("--max-per-category", type=int, default=3, help="Rows per category when --examples category is used.")
    parser.add_argument("--limit", type=int, help="Maximum total rows to export after filtering.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of ranked object candidates to show.")
    parser.add_argument(
        "--match-source",
        choices=["exact", "folder-first"],
        default="exact",
        help="exact embeds selected queries and uses the best retrieved gallery image; folder-first is a fast fallback.",
    )
    parser.add_argument("--top-k-images", type=int, default=DEFAULT_TOP_K_IMAGES, help="Image-level neighbors used to recover exact matched images.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--query-dir", type=Path, help="Override the query image folder.")
    parser.add_argument("--gallery-dir", type=Path, help="Override the gallery image folder.")
    parser.add_argument("--results-path", type=Path, help="Override the result CSV used for the selected variant.")
    parser.add_argument("--metadata-path", type=Path, help="Override the gallery metadata CSV used for the selected variant.")
    parser.add_argument("--embeddings-path", type=Path, help="Override the gallery embeddings .npy used for exact matched images.")
    parser.add_argument("--score-threshold", type=float, default=0.50, help="Displayed score threshold.")
    parser.add_argument("--margin-threshold", type=float, default=0.03, help="Displayed margin threshold.")
    return parser.parse_args()


def main() -> int:
    global QUERY_DIR, GALLERY_DIR

    args = parse_args()
    if args.query_dir:
        QUERY_DIR = args.query_dir
    if args.gallery_dir:
        GALLERY_DIR = args.gallery_dir

    variants = ["raw", "bg"] if args.variant == "both" else [args.variant]
    for variant in variants:
        spec = VARIANT_SPECS[variant]
        if args.results_path:
            spec["results"] = args.results_path
        if args.metadata_path:
            spec["metadata"] = args.metadata_path
        if args.embeddings_path:
            spec["embeddings"] = args.embeddings_path

    if not QUERY_DIR.exists() or not GALLERY_DIR.exists():
        print(
            f"Missing local data folder. Expected query images in {QUERY_DIR} and gallery images in {GALLERY_DIR}.",
            file=sys.stderr,
        )
        return 1

    generated_rows: list[dict[str, str]] = []
    exact_resolver = (
        ExactMatchResolver(top_k_images=args.top_k_images)
        if args.match_source == "exact"
        else None
    )

    for variant in variants:
        spec = VARIANT_SPECS[variant]
        results_path = spec["results"]
        metadata_path = spec["metadata"]
        if not results_path.exists():
            print(f"Skipping {variant}: missing {results_path}", file=sys.stderr)
            continue

        results_df = normalize_results_df(
            pd.read_csv(results_path),
            score_threshold=args.score_threshold,
            margin_threshold=args.margin_threshold,
        )
        known_truth_count = int(results_df["true_object"].apply(lambda value: clean_object_id(value) is not None).sum())
        print(f"{variant}: loaded {known_truth_count}/{len(results_df)} rows with known true_object labels.")
        gallery_lookup = build_gallery_lookup(metadata_path, GALLERY_DIR)
        selected_rows = select_rows(results_df, variant, args)
        if not selected_rows:
            print(f"No rows selected for {variant}.", file=sys.stderr)
            continue

        variant_dir = args.output_dir / variant
        for category, row in selected_rows:
            exact_matches = exact_resolver.resolve(row=row, variant=variant) if exact_resolver else {}
            generated_rows.append(
                make_card(
                    row=row,
                    variant=variant,
                    category_hint=category,
                    gallery_lookup=gallery_lookup,
                    exact_matches=exact_matches,
                    output_dir=variant_dir,
                    top_k=args.top_k,
                    score_threshold=args.score_threshold,
                    margin_threshold=args.margin_threshold,
                )
            )

    if not generated_rows:
        print("No visualizations were generated.", file=sys.stderr)
        return 1

    index_path = write_index(args.output_dir, generated_rows)
    print(f"Generated {len(generated_rows)} visualization(s).")
    print(f"Index: {index_path}")
    print(f"Visualizations: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
