# Hands-on Workshop Materials

This folder contains the simplified workshop version of the Step 1 image-retrieval prototype.

Participants build a small image-to-image retrieval system with:

- DINOv2 image embeddings
- FAISS nearest-neighbor search
- Top-1 and Top-3 result inspection
- confidence thresholds using score and margin
- prediction statistics and optional visualization cards

## Notebook

Run:

```text
notebooks/image_retrieval_hands_on.ipynb
```

The notebook is designed to be run top-to-bottom. It does not require the private prototype dataset in `step_1/data/`.

## Data Structure

Put gallery images into one folder per known object:

```text
data/
  gallery/
    object_001/
      img_001.jpg
      img_002.jpg
    object_002/
      img_001.jpg
      img_002.jpg
  query/
    query_001.jpg
    query_002.jpg
  query_labels.csv
  true_mapping.xlsx
```

Rules:

- Each folder inside `data/gallery/` is one searchable object.
- The folder name is the object ID, for example `object_001`.
- Put multiple reference views of the same object inside the same object folder.
- Put query images directly inside `data/query/`.
- Image filenames can be any valid image names.
- For labels, use either `true_mapping.xlsx` or `query_labels.csv`.
- If both files are present, the notebook uses `true_mapping.xlsx` first.

Example `query_labels.csv`:

```csv
query_name,true_object
query_001.jpg,object_001
query_002.jpg,object_002
query_unknown.jpg,
```

Leave `true_object` empty for unknown queries.

Alternative Excel label file:

```text
data/true_mapping.xlsx
```

The Excel file should have no required header row. The notebook reads the first two columns:

```text
query_001.jpg  object_001
query_002.jpg  object_002
query_003.jpg
```

It also supports numeric target IDs such as `66`, which are normalized to `object_066`.

## Optional Dataset Builder

The notebook includes an optional commented cell that can build a workshop dataset from the Amazon Berkeley Objects dataset using:

```text
../scripts/abo_retrieval_dataset_builder_v3.py
```

The builder can:

- filter objects by product type, for example `SOFA`
- download 360-degree image sequences
- create `data/gallery/object_###/` folders
- hold out one query image per object
- write `data/query_labels.csv`

Useful option:

```text
--max-gallery-images 4
```

This limits gallery views per object. For a 72-frame spin and `--max-gallery-images 4`, it selects frames `1, 24, 48, 72`.

The optional builder cell is commented out by default because it downloads external data and overwrites/adds files in `data/`.

## Main Workflow

The notebook performs these steps:

1. Load images from `data/gallery/` and `data/query/`.
2. Preview gallery and query images.
3. Load DINOv2.
4. Embed gallery images.
5. Build a FAISS index.
6. Search one query image.
7. Visualize Top-3 matches.
8. Search every query image.
9. Save prediction results.
10. Generate statistics tables and charts.

## Acceptance Rule

The notebook uses two thresholds:

```text
MIN_ACCEPT_SCORE = 0.50
MIN_ACCEPT_MARGIN = 0.03
```

A Top-1 object becomes the final prediction only if:

```text
top1_score >= MIN_ACCEPT_SCORE
margin >= MIN_ACCEPT_MARGIN
```

Otherwise the query is rejected as uncertain.

## Outputs

Generated files are written to `outputs/`:

- `gallery_embeddings.npy`
- `gallery_faiss.index`
- `gallery_metadata.csv`
- `query_results.csv`
- `prediction_statistics.csv`
- `prediction_statistics_by_object.csv`

`query_results.csv` includes:

- `top1_object`
- `top1_score`
- `second_score`
- `margin`
- `pred_object`
- `accepted`
- `top3_objects`
- `top3_scores`
- `top1_correct`
- `top3_correct`

## Statistics

The notebook creates summary tables and graphs for:

- accepted vs rejected predictions
- Top-1 accuracy on known queries
- Top-3 accuracy on known queries
- final decision accuracy
- Top-1 score distribution
- margin distribution
- hardest objects by accuracy

These outputs are useful for presentations and workshop discussion.

## Optional Visualizations

The notebook includes an optional commented cell that runs:

```text
../scripts/make_single_item_visualizations.py
```

It creates visual cards for the first 20 query results and saves them to:

```text
outputs/visualizations/
```

Each card shows:

- query image
- correct object image, if known
- Top-3 matched gallery images
- scores and margin
- final accept/reject decision

Run this only after `query_results.csv`, `gallery_metadata.csv`, and `gallery_embeddings.npy` have been generated.

## Rerun Guidance

If you change images or labels:

1. Rerun from **Load Images from the Workshop Folders**.
2. Rebuild the FAISS index.
3. Rerun **Search Every Query Image**.
4. Rerun **Prediction Statistics**.
5. Optionally regenerate visualization cards.
