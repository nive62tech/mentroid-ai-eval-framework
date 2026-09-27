# CV Sample Dataset

This is a small **synthetic** dataset (5 images, bounding boxes only — no
actual image files) used to demonstrate and smoke-test the CV evaluation
pipeline end-to-end without requiring a large download.

## Format

`ground_truth.json` maps `image_id -> list of ground-truth boxes`:

```json
{
  "img_001.jpg": [
    {"class": "car", "bbox": [x1, y1, x2, y2]}
  ]
}
```

- `bbox` is `[x1, y1, x2, y2]` in absolute pixel coordinates (top-left, bottom-right).
- `class` is the object category name.

Model predictions must be produced in the same shape, with an added
`"score"` (confidence) field:

```json
{
  "img_001.jpg": [
    {"class": "car", "bbox": [52, 58, 198, 182], "score": 0.93}
  ]
}
```

## Using your own dataset

To benchmark a real model, replace this data with your actual dataset:

1. Provide real image files (or however `predict_fn` expects to receive them —
   file paths, PIL images, tensors, etc.).
2. Provide a `ground_truth.json` in the format above, one entry per image.
3. Point `scripts/run_cv_eval.py --ground-truth <path> --images-dir <path>` at them
   (see the script's `--help` for all options).

### Requirements for a valid CV dataset
- At least one ground-truth box per relevant image (images with no objects are fine
  and simply contribute to false-positive counting).
- Consistent class naming between ground truth and model predictions.
- Enough images (recommended 50+) for stable Precision/Recall/mAP estimates;
  this 5-image sample is for pipeline verification only, not real model comparison.
