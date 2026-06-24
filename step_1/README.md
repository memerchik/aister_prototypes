# Step 1 - DINOv2-based object retrieval

This folder contains the actual notebook workflow for the image-retrieval prototype described in the root [README.md](../readme.md).

## What is here

- `notebooks/step_1-01_dinov2_image_retrieval.ipynb`: main technical notebook
- `notebooks/step_1-02_dinov2_workshop_tutorial.ipynb`: workshop version with navigation, teaching notes, live statistics, charts, and exercises
- `notebooks/step_1-03_dinov2_image_retrieval_background_removal.ipynb`: separate background-removal experiment
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

1. [Background removal notebook](../readme.md#background-removal-notebook)
   This explains when to use the separate background-removal notebook and cache.
2. [Run with gallery](../readme.md#run-with-gallery-using-predefined-best-parameters)
   This rebuilds embeddings, metadata, and the FAISS index.
3. [Run with new best parameters](../readme.md#run-with-new-best-parameters)
   This retunes the score and margin thresholds after rebuilding or updating the gallery.
4. [Run from cache](../readme.md#run-from-cache)
   This reuses the current cached gallery artifacts and can work without the gallery folder.

## Main outputs

- `outputs/gallery_embeddings.npy`
- `outputs/gallery_faiss.index`
- `outputs/gallery_index_metadata.csv`
- `outputs/metrics_fixed.json`
- `outputs/query_results_fixed.csv`

The background-removal notebook writes equivalent files with `_bg_removed` in the filename.

For the fuller repository context, data layout explanation, and current metrics summary, use the root [README.md](../readme.md).
