# Step 1 - DINOv2-based object retrieval

This folder contains the actual notebook workflow for the image-retrieval prototype described in the root [README.md](../readme.md).

## What is here

- `notebooks/step_1-01_dinov2_image_retrieval.ipynb`: main raw-image retrieval notebook
- `notebooks/step_1-02_dinov2_image_retrieval_background_removal.ipynb`: separate background-removal experiment
- `scripts/make_single_item_visualizations.py`: helper for generating single-item Top-3 retrieval visualizations from saved results
- `workshop_materials/`: simplified hands-on workshop notebook and local data template for participants
- `outputs/`: cached embeddings, FAISS index, metadata, metrics, and query results
- `data/`: local-only input data required to run the notebook

## What the notebook expects

The notebook expects local input files under `step_1/data/`.
For the full structure, mapping examples, and operational details, use the root [README.md](../readme.md#reproducibility-notes).

- `data/` is local-only and should not be published to GitHub.
- The notebook expects `data/dataset_dev/query/` and `data/true_mapping.xlsx` in all modes.
- `data/dataset_dev/gallery/` is required only when rebuilding the gallery cache.
- Background removal lives in its own notebook. Use the root [background removal notes](../readme.md#background-removal-notebook) before running that experiment.

## Run

Use these links to run this step:

1. [Hands-on workshop materials](workshop_materials/README.md)
   This is the simplified participant workflow for learning the technology and trying new images.
2. [Background removal notebook](../readme.md#background-removal-notebook)
   This explains when to use the separate background-removal notebook and cache.
3. [Run with gallery](../readme.md#run-with-gallery-using-predefined-best-parameters)
   This rebuilds embeddings, metadata, and the FAISS index.
4. [Run with new best parameters](../readme.md#run-with-new-best-parameters)
   This retunes the score and margin thresholds after rebuilding or updating the gallery.
5. [Run from cache](../readme.md#run-from-cache)
   This reuses the current cached gallery artifacts and can work without the gallery folder.

## Single Item Visualizations

Generate local PNG visualizations that show a query image, the correct object image for known queries, Top-3 object candidates, score/margin values, and the final decision:

```bash
python step_1/scripts/make_single_item_visualizations.py
```

By default, the helper embeds only the selected query images and compares them with the cached gallery embeddings so each Top-3 visualization shows the exact retrieved gallery image. Use `--match-source folder-first` only as a faster fallback when you do not need exact matched images.

Useful variants:

- `python step_1/scripts/make_single_item_visualizations.py --variant raw --queries img_030.JPG img_077.png`
- `python step_1/scripts/make_single_item_visualizations.py --examples category --categories wrong_approval false_unknown_approval`
- `python step_1/scripts/make_single_item_visualizations.py --variant bg --score-threshold 0.54 --margin-threshold 0.015`
- `python step_1/scripts/make_single_item_visualizations.py --match-source folder-first`

Visualizations are written to `outputs/single_item_visualizations/`. This folder is git-ignored because the generated PNGs contain local dataset images.

## Main outputs

- `outputs/gallery_embeddings.npy`
- `outputs/gallery_faiss.index`
- `outputs/gallery_index_metadata.csv`
- `outputs/metrics_fixed.json`
- `outputs/query_results_fixed.csv`

The background-removal notebook writes equivalent files with `_bg_removed` in the filename.

For the fuller repository context, data layout explanation, and current metrics summary, use the root [README.md](../readme.md).
