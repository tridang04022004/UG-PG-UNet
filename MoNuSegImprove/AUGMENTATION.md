AUGMENTATION (based on train/aug.py)

This document summarizes how dataset augmentation is implemented in `MoNuSegImprove/train/aug.py`.

Overview
--------
The augmentation pipeline does the following:

- Reads images (`*.tif`) from `MoNuSegImprove/train/images` and annotations from `MoNuSegImprove/train/annots`.
- For each image, creates an optional raster mask from either a matching mask file (same filename in `annots`) or from XML polygon annotations (`.xml`).
- Extracts overlapping square patches from the image and mask (default 256×256 patches with 128 px stride).
- Saves each patch (image and mask) to `MoNuSegImprove/train/aug/images` and `MoNuSegImprove/train/aug/annots`.
- If XML polygon regions exist, crops polygon coordinates to each patch and writes a patch-level XML with the polygons that intersect the patch.
- Applies multiple augmentations per patch using `albumentations` (default 3 augmentations per patch), optionally transforming polygon keypoints so a new patch-level XML is written for each augmented sample.

Key parameters
--------------
- `PATCH_SIZE` (default 256): size of square patches extracted from images.
- `STRIDE` (default 128): spacing between patch top-left corners (overlap = PATCH_SIZE - STRIDE).
- `AUG_PER_PATCH` (default 3): number of augmented variants generated per patch.

Transforms (albumentations)
---------------------------
The script uses an `albumentations.Compose` with these transforms (with probabilities):

- HorizontalFlip (0.5)
- VerticalFlip (0.5)
- RandomRotate90 (0.5)
- RandomBrightnessContrast (0.4)
- HueSaturationValue (0.3)
- ElasticTransform (alpha=50, sigma=5, p=0.3)
- GridDistortion (0.3)
- GaussianBlur (0.2)
- GaussNoise (0.2)

Polygons are converted into `keypoints` for albumentations so geometric transforms (flip/rotate/elastic/grid) keep polygon alignment with masks.

XML handling
------------
- Input XML shape: the script expects the polygon structure used in MoNuSeg (Annotations -> Annotation -> Regions -> Region -> Vertices -> Vertex with X/Y attributes).
- `xml_to_mask(xml_path, image_shape)` rasterizes all polygons into a binary mask (255 inside polygons, 0 otherwise) using OpenCV `fillPoly`.
- `xml_to_regions(xml_path)` returns a list of regions where each region is a list of (x, y) float coordinates in image space.
- `regions_to_xml(regions, out_path)` writes a minimal XML with polygons for a patch-level file. Coordinates are stored as floats with 6 decimal places.
- When cropping polygons to a patch, only vertices that lie within the patch are kept; a region is saved only if it has at least 3 vertices after cropping.

Outputs
-------
- Augmented images are written to `MoNuSegImprove/train/aug/images`.
- Augmented masks (rasterized) and patch-level XMLs are written to `MoNuSegImprove/train/aug/annots`.
- Filenames follow the convention: `{image_stem}_{patch_index}.tif`, and augmented variants `{image_stem}_{patch_index}_aug{k}.tif`. XMLs use `.xml` extension and same stem.

Usage
-----
Run the script from the `MoNuSegImprove/train` directory (script uses local relative paths):

```powershell
cd MoNuSegImprove\train
python aug.py
```

Notes and caveats
-----------------
- The script depends on `albumentations` and `opencv-python` (`cv2`). Install with:

```powershell
pip install albumentations opencv-python
```

- Large images and many patches can produce many output files — ensure sufficient disk space.
- The polygon cropping logic discards polygon vertices that fall outside the patch; it does not attempt to split or clip polygons at patch boundaries. This means that polygons partially intersecting a patch but with fewer than 3 vertices inside the patch will be dropped for that patch.
- The script treats masks and XMLs separately: if a raster mask file with the same image name exists in `annots`, it will be used; otherwise the script tries to rasterize the `.xml` annotation.

Implementation pointers (from `aug.py`)
--------------------------------------
- `xml_to_mask` uses `ElementTree` to parse XML and `cv2.fillPoly` to rasterize polygons.
- Patches are enumerated by scanning a grid computed from image dimensions, patch size and stride.
- When regions are present, they are flattened into a single keypoints list for `albumentations` and reconstructed after augmentation using stored split lengths.

If you'd like, I can also:
- Add a CLI wrapper to configure `PATCH_SIZE`, `STRIDE`, `AUG_PER_PATCH` from command-line args.
- Modify polygon cropping to clip polygon edges and produce valid polygons from intersections (instead of dropping regions with <3 vertices).

---
Document generated from `MoNuSegImprove/train/aug.py` (Oct 16, 2025).