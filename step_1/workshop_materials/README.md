# Hands-on Workshop Materials

This folder contains a simple notebook for recreating the image-retrieval prototype with a small image set.

## What Participants Will Build

Participants will:

- organize a gallery of known objects,
- add query images,
- turn images into DINOv2 embeddings,
- build a FAISS similarity index,
- search for the closest gallery objects,
- inspect Top-3 results visually.

## Correct Data Storage

Put gallery images into one folder per object:

```text
data/gallery/
  object_001/
    img_001.jpg
    img_002.jpg
  object_002/
    img_001.jpg
```

Put query images directly into `data/query/`:

```text
data/query/
  query_001.jpg
  query_002.jpg
```

If you know the correct answer for a query, add it to `data/query_labels.csv`:

```csv
query_name,true_object
query_001.jpg,object_001
query_002.jpg,object_002
query_unknown.jpg,
```

Leave `true_object` empty for unknown queries.

## Main Notebook

Run:

```text
notebooks/image_retrieval_hands_on.ipynb
```

The notebook is designed to be run top-to-bottom. It does not require the private prototype dataset in `step_1/data/`.

## Outputs

Generated files are written to `outputs/`:

- `gallery_embeddings.npy`
- `gallery_faiss.index`
- `gallery_metadata.csv`
- `query_results.csv`
