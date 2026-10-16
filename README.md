# UltrAi Nerve Viewer

A desktop PyQt6 application for nerve ultrasound. It loads images and videos, annotates nerves and other anatomy as labelled instance masks, segments them with ONNX models, and measures the area of each mask.

## Features

- **Files:** images (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`), videos (`.mp4`, `.avi`, `.mov`, `.mkv`, `.m4v`, `.wmv`) and multi-page TIFF stacks.
- **Labels:** the scanned location of each file, plus the nerve or other-anatomy label of each mask, picked from chips. Custom labels can be added.
- **Instance masks:** several masks per image or frame, each tagged on the canvas, for example "Median #1". A mask can be relabelled or deleted from its right-click menu.
- **Editing tools:** freehand and segmented lines, paint brush, eraser, undo and redo.
- **Segmentation:** ONNX models segment the image on screen, a whole video, or every loaded file. Segmentation can be confined to an ROI box, and Auto ROI finds the ultrasound field of view.
- **Rotation:** each file can be viewed rotated without changing the file itself.
- **Measurements:** the area of every mask, in mm² once calibrated, together with the file's side (left/right) and location.
- **Hardware:** runs on the CPU, or on CUDA through ONNX Runtime when available.

## Output layout

Masks use the same layout as the web annotator (`web_based_app` branch), so either app opens the other's output:

```text
<output folder>/
  nerve_manifest.json            label sets; location and side of each file
  rois.csv, rotations.csv        ROI box and rotation of each file
  measurements_<model>.csv       one row per mask: label, area, side, location
  <model>_predictions/
    <image stem>/median_001.png
    <video stem>/frame_000012_median_001.png
```

With **No model** selected, the masks go in the output folder itself and the measurements in `measurements.csv`.

## Models

- **Location:** every `.onnx` file in `assets/` is listed under **Model**.
- **One-channel input:** the model is fed a grey image, min-max scaled to 0–1 and resized to the model's input.
- **Three-channel input:** the model is treated as a DINOv3 model (as trained in `fm_adaptation`). It is fed the RGB image, resized with its aspect ratio kept onto the model's square input and normalised with the ImageNet mean and standard deviation. If the input size is not fixed, 896 is used.
- **Output:** the largest connected piece of each class becomes a mask.
- **Labels:** a single-class model's masks get the nerve picked in the sidebar, or `median`. A model with several classes can name them in a `<model>.json` beside it:

  ```json
  {"classes": ["background", "median", "artery"]}
  ```

- **Replacement:** a model's masks replace saved masks of the same labels and leave other labels alone.

## Running

```bash
pip install -r requirements.txt
python app.py
```

## Build

- **Linux:** `./build_linux.sh` builds a PyInstaller bundle in `dist/`, and `./ship.sh` packs it into a tarball.
- **Windows:** `build_windows.bat` builds the bundle and, when Inno Setup 6 is installed, an installer from `UltrAiViewerInstaller.iss`. `ship.bat` zips it.

Models must be in `assets/` at build time. Inference logs are written to `inference.log`.

See [`userguid.md`](userguid.md) for the user guide.
