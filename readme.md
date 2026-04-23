# AISTER Prototypes

## Case Study 1: DINOv2 Image Retrieval

### Overview

The `dinov2_prototype.ipynb` notebook implements an **image-based object retrieval system** that uses the pretrained DINOv2 vision model to match query images against a gallery of known objects.

### What It Does

This notebook solves an **object identification task**: given a query image, determine which object from a gallery it belongs to, or classify it as unknown. It combines:

- **Feature Extraction**: Uses DINOv2 (a self-supervised vision model) to extract semantic embeddings from images
- **Fast Similarity Search**: FAISS enables quick retrieval of the most similar gallery images
- **Intelligent Aggregation**: Groups matches by object and computes confidence scores
- **Threshold-Based Confidence**: Uses dual thresholds (score and margin) to decide when predictions are reliable enough

### How to Use It

1. **Prepare your data**:
   - Organize gallery images in `data/dataset_dev/gallery/object_001/`, `object_002/`, etc.
   - Place query images in `data/dataset_dev/query/`
   - Create a `data/true_mapping.xlsx` file with two columns: query identifier → target object (or "none")
   
   **Example mapping format (Excel sheet .xlsx; ignore the column names and numeration. You sheet must have only 2 columns without naming. 1st - query image name; 2nd - object's folder name or none):**
   | -   | A           | B          |
   | --- | ----------- | ---------- |
   | 1   | query_img_1 | object_001 |
   | 2   | query_img_2 | object_005 |
   | 3   | query_img_3 | none       |
   | 4   | query_img_4 | object_012 |

2. **Run the notebook**:
   - Execute all cells from top to bottom
   - First run builds embeddings and FAISS index (takes time), subsequent runs use cache

3. **Outputs generated**:
   - `dinov2_prototype_output/gallery_embeddings.npy` - Cached embeddings
   - `dinov2_prototype_output/gallery_faiss.index` - FAISS index for fast search
   - `dinov2_prototype_output/metrics_fixed.json` - Evaluation metrics
   - `dinov2_prototype_output/query_results_fixed.csv` - Per-query predictions and scores

4. **Inspect results**:
   - Use the `inspect_result_row()` function to visualize a query, its matches, and top-ranked objects
   - Check `final_results_df` for prediction errors and investigate misclassifications

### Pipeline Diagram

![DINOv2 Flowchart](case-study-1/DINOv2%20Flowchart.png)