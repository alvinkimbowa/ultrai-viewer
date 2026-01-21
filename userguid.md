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
2) Load an image: File > Load Image.
3) Select a model (left panel, Model dropdown).
4) Click Segment to run inference.
5) Edit results if needed (Freehand Line, Segmented Line, Paint Brush, Eraser).
6) Save the mask: File > Save Mask (PNG recommended).

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

1) Load an image.
2) Choose the model.
3) Click Segment.
4) Wait for the progress dialog to finish.
5) Edit the mask if needed.

## Batch Segmentation (Many Images)

1) Click "Load image sequence".
2) Select a folder or select multiple images.
3) Choose an output folder.
4) Click Batch segment.
5) Wait for the progress dialog to finish.

Results are saved automatically in the output folder as PNG files. You can preview
results by navigating through the loaded images using the left/right arrow keys.

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
- Scroll: vertical scroll (when zoomed).

## Saving

- Save Mask: saves the current mask as PNG.
- If a batch output folder is selected, Ctrl+S saves directly to that folder.

## Tips

- If the image looks too dark or bright, check the original export.
- Large images may take longer to process.
- Clean ROI crops improve stability.

## Troubleshooting

- "Unsupported image layout": the file is not a standard 2D or RGB image.
- If batch results are empty, check image quality and orientation.
