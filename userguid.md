# UltrAi Nerve Viewer User Guide

Version 0.1.0 (beta). For research use only; results need clinical validation.

## Install

- **Windows:** run `UltrAiNerveViewer-v0.1.0-windows-x64-installer.exe`, then open **UltrAi Nerve Viewer** from the Start menu or the desktop.
- **Linux:** run `UltrAiNerveViewer-v0.1.0-linux-x86_64` in the folder this guide is in.

Runs on the CPU; a 4-core CPU and 8 GB of RAM are recommended. macOS is not supported.

## Load and label

1. Click **Load files** and choose images (PNG, JPEG, TIFF, BMP) or videos, and an output folder.
2. Under **Labels**, pick the **Location** the file was scanned at. Use **+ Add location** for a site that is not listed.
3. Pick the **Nerve** or **Other anatomy** label that new masks get. **+ Add nerve** and **+ Add anatomy** add labels of your own.

Scroll, use the arrow keys, or **Prev** / **Next** to move between images and frames.

## Draw and edit masks

- **Freehand Line** and **Segmented Line** draw an outline. With **Fill mask** ticked, each outline becomes a new mask of the picked label.
- **Paint Brush** paints into the selected mask, or starts a new one; **Eraser** erases from every mask.
- Each mask is tagged with its label and number, such as **Median #1**.
- Right-click a mask to change its **Label** or to **Delete mask**. You can also click it with **Select** and press Delete.
- **Delete masks** in the sidebar deletes every mask of the image or frame on screen.
- Ctrl+Z and Ctrl+Y undo and redo.

## Segment with a model

1. Choose a model under **Model**.
2. Click **Segment image** for the image or frame on screen, **Segment video** for the whole video, or **Segment all files** for everything loaded.

A model that segments one structure labels its masks with the nerve picked, or **Median** when none is picked. The model's masks replace masks of the same label; masks of other labels stay.

## Measurements

1. **Calibrate:** enter **Pixels per mm**, or click **Calibrate** and drag the lines onto a known distance. Until then, areas are in pixels.
2. **Side:** choose **Left** or **Right** for the file on screen.
3. **Read the results** in the **Measurements** panel: the area of each mask on screen.

## Where results are saved

Everything is saved automatically as you go:

- **Masks:** `<output folder>/<model_name>_predictions/<file name>/`, one PNG per mask.
  - Images: `median_001.png`.
  - Videos: `frame_000012_median_001.png`.
  - With **No model** selected, masks go in the output folder itself.
- **Measurements:** `<output folder>/measurements_<model_name>.csv`, one row per mask with its label, area, side and location.
- **Labels:** `<output folder>/nerve_manifest.json` holds the label lists and each file's location and side.

## Tips

- The models are most reliable when they see only the ultrasound field of view and the image is upright:
  - **Auto ROI** boxes the field of view on the current image or on all images. You can also draw a box with **ROI Box** under **Tools**. Boxes can be moved, resized or deleted.
  - **Rotate image** straightens a tilted image. The image file itself is not changed.
