# UltAI Viewer

Desktop PyQt6 application for reviewing knee ultrasound data, running ONNX cartilage segmentation, editing masks manually, and organizing video annotations.

## What This Repo Contains

- A desktop viewer and annotation tool launched from `app.py`
- ONNX Runtime integration for single-image and batch image segmentation
- Manual mask editing tools: select, freehand line, segmented line, paint brush, eraser
- Image-sequence workflow for batch segmentation and mask export
- Video workflow for frame-by-frame annotation with per-video nerve/location labels
- PyInstaller packaging for Linux and Windows

## Features

- Load images: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`
- Load videos: `.mp4`, `.avi`, `.mov`, `.mkv`, `.m4v`, `.wmv`
- Run segmentation on the currently loaded image
- Run batch segmentation across an image sequence
- Edit masks with undo/redo
- Adjust mask visibility, opacity, and brush radius
- Save masks as `.png` or `.tif/.tiff`
- Resume video labeling from previously saved frame masks
- Persist video metadata in `nerve_manifest.json`
- Use CPU by default, with CUDA ONNX Runtime support when available

## Repository Layout

- [`app.py`](/home/ultrai/UltrAi/knee_cat_seg/app.py): app entrypoint
- [`src/main_window.py`](/home/ultrai/UltrAi/knee_cat_seg/src/main_window.py): main UI and workflow logic
- [`src/canvas.py`](/home/ultrai/UltrAi/knee_cat_seg/src/canvas.py): image display and mask editing
- [`src/model_integration.py`](/home/ultrai/UltrAi/knee_cat_seg/src/model_integration.py): ONNX model loading and inference
- [`assets/`](/home/ultrai/UltrAi/knee_cat_seg/assets): bundled model(s) and UI icons
- [`UltrAiViewer.spec`](/home/ultrai/UltrAi/knee_cat_seg/UltrAiViewer.spec): PyInstaller spec
- [`build_linux.sh`](/home/ultrai/UltrAi/knee_cat_seg/build_linux.sh): Linux packaging
- [`build_windows.bat`](/home/ultrai/UltrAi/knee_cat_seg/build_windows.bat): Windows packaging

## Requirements

- Python 3.10+ recommended
- Linux or Windows
- An ONNX model in `assets/`

The repo currently includes:

- `assets/nnunet_200.onnx`

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Running The App

From the repository root:

```bash
python app.py
```

## Typical Workflows

### Single image segmentation

1. Launch the app.
2. Load one or more images.
3. Select a model from the sidebar.
4. Click `Segment`.
5. Refine the result with the editing tools if needed.
6. Save the mask.

### Batch image segmentation

1. Load an image sequence and choose an output folder.
2. Select a model.
3. Click `Batch segment`.
4. Masks are written to the chosen output folder as `<image_stem>.png`.

### Video annotation

1. Load one or more videos and choose an output folder.
2. Navigate across videos and frames.
3. Draw or edit masks frame by frame.
4. Assign `Location` and `Nerve` labels.
5. Save masks to persist annotated frames.

For video mode, masks are stored under:

```text
<output_dir>/
  nerve_manifest.json
  <video_stem>/
    frame_000000.png
    frame_000001.png
    ...
```

`nerve_manifest.json` stores the available label sets and the per-video selected labels.

## Controls And Shortcuts

- `Ctrl+S`: save mask
- `Ctrl+Z`: undo
- `Ctrl+Y`: redo
- `Delete`: clear mask
- `P`: previous image/video item
- `N`: next image/video item
- Left/Right arrows in video mode: move across frames
- Mouse wheel + modifiers: zoom/scroll in the canvas

## Model Behavior

- Models are discovered from `assets/` with `.onnx` or `.nnx` extensions.
- Input images are normalized to grayscale before inference.
- Output masks are postprocessed to keep the largest connected component.
- If GPU execution is selected but unavailable, the app falls back to CPU.

## Build

### Linux

```bash
./build_linux.sh
```

This builds a PyInstaller bundle in `dist/` and copies the PDF user guide into the packaged app directory.

To create a tarball for distribution:

```bash
./ship.sh
```

### Windows

```bat
build_windows.bat
```

This builds the PyInstaller output in `dist\`. If Inno Setup 6 is installed, the script also creates a Windows installer using [`UltrAiViewerInstaller.iss`](/home/ultrai/UltrAi/knee_cat_seg/UltrAiViewerInstaller.iss).

To create a zip archive after building:

```bat
ship.bat
```

## Notes

- Batch segmentation currently applies to image sequences, not videos.
- Video mode automatically saves the current frame mask when you move between frames.
- The packaged build includes icons and `VERSION`; models must still be present in `assets/` at build time.
- Inference logs are written to `inference.log`.

## Citation

If you use this software in research, cite the repository and the specific version or GitHub release you used.

Suggested citation:

```text
UltAI Viewer (knee_cat_seg), version 0.2.1. GitHub repository:
https://github.com/alvinkimbowa/ultrai-viewer
```

Example BibTeX:

```bibtex
@software{knee_cat_seg_2026,
  title = {UltAI Viewer},
  version = {0.2.1},
  url = {https://github.com/alvinkimbowa/ultrai-viewer},
  note = {GitHub repository for knee ultrasound cartilage segmentation and annotation software}
}
```

## Related Files

- [`userguid.md`](/home/ultrai/UltrAi/knee_cat_seg/userguid.md): lightweight user-facing guide
- [`User Guide - UltrAi Viewer.pdf`](/home/ultrai/UltrAi/knee_cat_seg/User%20Guide%20-%20UltrAi%20Viewer.pdf): packaged PDF guide
