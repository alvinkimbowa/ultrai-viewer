# UltAI Viewer User Guide

This guide covers UltAI Viewer for knee ultrasound cartilage segmentation.

## App Version

Current version: 0.1.0

## Notes and Disclaimers

- This is a beta release; features and results may change.
- Results are for research use only and require clinical validation.

## Supported OS

- Windows
- Linux

macOS is not supported at this time.

## App Size

The current app size is ~800 MB. Future versions will be smaller and more optimized.

## Compute Requirements

- CPU-only build.
- Recommended: modern 4+ core CPU and 8+ GB RAM.

## Quick Start

1) Launch the app (UltAIViewer.exe on Windows, UltAIViewer on Linux).
2) Click **Load files**, choose your images and an output folder.
3) Select a model (left panel, Model dropdown).
4) Click **Segment image** to run inference.
5) Edit results if needed (Freehand Line, Segmented Line, Paint Brush, Eraser).
6) Click **Save masks** (or press Ctrl+S).

## Supported Images and ROI

- Image type: 2D knee ultrasound images.
- Preferred format: grayscale TIFF. PNG, JPEG, and BMP are also supported.
- Recommendation: use clean, well-cropped ROI images for best segmentation.

## Models and Speed

- Lean model: faster, smaller.
- Full model: more accurate, higher CPU load.
- Batch segmentation runs on CPU only.

## Device Selection

This build runs on CPU only.

## Segmentation (Single Image)

1) Click **Load files** and choose the image(s) and output folder.
2) Choose the model.
3) Click **Segment image**.
4) Wait for the progress dialog to finish.
5) Edit the mask if needed.

A load is either images or videos, not both together. **Segment video** is only
available while videos are loaded.

## Batch Segmentation (Many Images)

1) Click **Load files**.
2) Select a folder or select multiple images.
3) Choose an output folder.
4) Click **Segment all files**.
5) Wait for the progress dialog to finish.

Results are saved automatically in the output folder as PNG files. You can preview
results by navigating through the loaded images using the left/right arrow keys.
If saved masks already exist, the app asks whether to overwrite them or preserve
them and segment only missing images.

## Video Segmentation

1) Click **Load files** and choose the videos and output folder.
2) Use **Segment image** to segment only the displayed frame for editing.
3) Use **Segment video** to segment every frame of the selected video.
4) Use **Segment all files** to segment every frame of all loaded videos.

If saved masks already exist, the app asks whether to overwrite them or preserve
them and segment only missing frames. Masks are saved as PNG files under
`<output folder>/<video name>/frame_000000.png`. Canceling retains frames already
processed and saved.

## Editing Tools

- Freehand Line: draw an outline ROI.
- Segmented Line: click to add points, right-click or double-click to finish.
- Paint Brush: add mask.
- Eraser: remove mask.
- Tool Radius: controls edit thickness.
- Fill ROI: unchecked shows contour; checked fills the ROI into the mask.
- Ctrl+Z to undo, Ctrl+Y to redo.

## Zoom and Scroll

- Ctrl + scroll: zoom in/out.
- Shift + scroll: horizontal scroll (when zoomed).
- Scroll up/down: move to the previous/next image or video frame.
- Prev/Next: move to the previous/next image, or the previous/next video.

## Saving

- Save masks (Ctrl+S): saves the current mask as PNG in the output folder.
- Select output folder: switches to a different output folder without reloading the files.
- Delete mask: removes the mask from the screen and deletes its saved file from the output folder. Ctrl+Z brings the mask back on screen.
- Close file: closes the image or video on screen and keeps the others loaded.

## Tips

- If the image looks too dark or bright, check the original export.
- Large images may take longer to process.
- Clean ROI crops improve stability.

## Troubleshooting

- "Unsupported image layout": the file is not a standard 2D or RGB image.
- If batch results are empty, check image quality and orientation.
