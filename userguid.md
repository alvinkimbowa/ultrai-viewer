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

Results are saved automatically as PNG files in the selected model's `<model>_predictions` subfolder of the output folder. You can preview
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
`<output folder>/<model>_predictions/<video name>/frame_000000.png`. Canceling retains frames already
processed and saved.

## Editing Tools

- Freehand Line: draw an outline.
- Segmented Line: click to add points, right-click or double-click to finish.
- Paint Brush: add mask.
- Eraser: remove mask.
- Tool Radius: controls edit thickness.
- Fill mask: unchecked shows only the outline; checked fills the outline into the mask.
- ROI Box: draw the ROI (see ROI below).
- Delete: right-click the mask, the ROI or an unfilled outline and choose Delete; or click the mask or the ROI with the Select tool and press the Delete key.
- Ctrl+Z to undo, Ctrl+Y to redo. Undo and Redo cover mask edits and ROI changes (drawing, moving, resizing, deleting a box) on the image on screen, in the order you made them. This history starts fresh when you move to another image or frame.
- Undo also takes back **Apply ROI to all** and **Clear all ROIs**, even after you have moved to another image or frame: the earlier ROIs, the masks as they were before trimming and their measurements are put back. An image you edited after the apply is left as it is.

## Rotate Image

**Rotate image** (Edit section, or Edit menu) turns the image on screen, for example to straighten a tilted scan. The image file itself is never changed.

- A box opens with a slider from −180° to 180° and a number box beside it for steps of 0.1°. The image and its mask turn as you move the slider. Positive angles turn the image clockwise.
- **OK** keeps the angle, **Cancel** puts the image back as it was, **Reset** sets 0°.
- The rotation belongs to that file: each image has its own, and a video has one for all its frames. It is kept in `rotations.csv` in the output folder, so the file opens rotated the next time.
- Everything then works on the rotated view: segmentation, the ROI, the regions and all measurements, including Pixels per mm x and y. If x and y are different, calibrate on the rotated view.
- Mask files are still saved to line up with the original, unrotated image file. A mask edited on a rotated view can shift by about a pixel at its edge when saved, and anything drawn in the black corners is not saved.
- Changing a file's rotation removes its ROI, since the view changes size as the image turns and the box no longer covers the same part. Its saved measurements are worked out again at the new angle.

## Zoom and Scroll

- Ctrl + scroll: zoom in/out.
- Shift + scroll: horizontal scroll (when zoomed).
- Scroll up/down: move to the previous/next image or video frame.
- Prev/Next: move to the previous/next image, or the previous/next video.

## Saving

Results are kept per model, inside the output folder you select. A model's masks go in a subfolder named `<model>_predictions`, and its measurements go in a file named `measurements_<model>.csv` beside it:

```
<output folder>/
    rois.csv
    rotations.csv
    measurements_nnunet-D1.csv
    nnunet-D1_predictions/
        image1.png
        <video name>/frame_000000.png
    measurements_monounet202.csv
    monounet202_predictions/
        ...
```

- Changing the **Model** changes where results are saved and switches the mask on screen to the one saved for that model (or to no mask, if it has none for this image or frame). Anything you changed is first saved under the model you are leaving.
- **No model** in the Model list shows and saves masks directly in the output folder, with no subfolder, and their measurements in `measurements.csv`. The Segment buttons need a model.
- Results that sit directly in the output folder (saved before models had subfolders) are shown when the selected model has no mask of its own for that image or frame. They are not rewritten: if you edit one, the edited mask is saved in the model's `<model>_predictions` subfolder and that copy is shown from then on. **Delete mask** does delete the one being shown.
- Save masks (Ctrl+S): saves the current mask as PNG in the selected model's `<model>_predictions` subfolder.
- Leaving an image or frame saves its mask and measurements only if something changed (the mask, ROI, centre point or knee side). Just looking at it writes nothing, except that a mask with no measurements yet gets them.
- Changing Pixels per mm updates the saved measurements of the image on screen, or of every saved frame of the video on screen. Other files keep theirs until they are edited or saved.
- Select output folder: switches to a different output folder without reloading the files.
- Delete mask: removes the mask from the screen and deletes its saved file. Ctrl+Z brings the mask back on screen.
- Close file: closes the image or video on screen and keeps the others loaded.

## ROI

An ROI is a box you draw on an image so that only the part inside it is segmented. Each image or video frame has its own ROI, or none.

- **Draw**: choose **ROI Box** in the Tools dropdown and drag a box on the image. To replace a box, start the drag outside it.
- **Apply to others**: after you draw, the app asks whether to apply the box to all the other loaded images and frames. Yes replaces whatever ROI they had; where one of them already has a saved mask under the selected model, that mask is trimmed to the box straight away. No keeps the box on this image or frame only. Cancel discards the box you just drew and puts back the ROI the image had before, if any. Afterwards each image or frame can still be given a different ROI of its own.
- **Adjust**: with the ROI Box or the Select tool, drag an edge or a corner to resize the box, or drag inside it to move it. This changes the ROI of that image or frame only.
- **Delete**: right-click the box and choose **Delete ROI**, or click it and press the Delete key. Only that image or frame loses its ROI.
- **Apply ROI to all** (right-click a box): gives the box, as it is now, to all the other loaded images and frames, after asking. Use it after moving or resizing a box. It does the same as answering Yes right after drawing.
- **Clear all ROIs** (Edit menu, or right-click a box): removes the ROI from every loaded image and frame, and from their rows in the selected model's measurements file. Saved masks are not changed.
- Segment image, Segment video and Segment all files use each image's or frame's own ROI, and the whole image where there is none.
- A mask that already exists is hidden outside the box. Only the part inside is saved and measured. The hidden part comes back if you enlarge or delete the box before leaving that image or frame.
- ROIs are kept in `rois.csv` in the output folder, shared by all models, so they are there the next time you open the same files.
- Once a model has a saved mask for an image or frame, the ROI recorded with it in that model's measurements file is the one shown and used under that model. Changing the ROI under one model does not change what another model has saved; `rois.csv` supplies the ROI only where the selected model has nothing saved yet.

## Measurements

The Measurements section of the left panel shows numbers for the mask on screen. They update by themselves when the mask changes.

- **Pixels per mm (x, y)**: the image scale. Type the values, or click **Calibrate**, drag the two horizontal and two vertical lines onto a known distance, and enter the real distance between each pair. Without both values, results are shown in pixels.
- **View**: **Full cartilage** measures the mask as one piece. **Per region** splits it into lateral, notch (the intercondylar region) and medial, each in its own colour.
- **Knee**: Right or Left. It decides which side of the image is lateral and which is medial, and stays as set for the following images or frames until you change it.
- **Centre point**: in Per region view a yellow diamond marks the middle of the intercondylar notch on the top surface of the cartilage. The app suggests a position; with the Select tool, drag it left or right to correct it. **Reset centre point** returns to the suggestion. The notch region is centred on the diamond and is 25% of the width of the ROI, or 25% of the image width when the image has no ROI.
- **Area**: size of the mask.
- **Length**: length of the cartilage-bone line, shown as a dotted line. The line is the bottom side of the mask's outline, from the leftmost to the rightmost point of the mask, and the length is the shortest distance along it between those two ends, following every bend. A mask in separate pieces is measured piece by piece and added up. In Per region view, each region gets the part of the line that lies inside it.
- **Thickness**: area divided by length.
- **Echo intensity** and **Variation**: the average brightness inside the mask and how much it varies (standard deviation), in arbitrary units (AU).

Every time a mask is saved, its measurements are saved to the selected model's `measurements_<model>.csv` in the output folder, one row per image or video frame, holding both the whole-mask and the per-region numbers. The `rotation` column records the angle the image was turned by when it was measured.

## Tips

- If the image looks too dark or bright, check the original export.
- Large images may take longer to process.
- Clean ROI crops improve stability.

## Troubleshooting

- "Unsupported image layout": the file is not a standard 2D or RGB image.
- If batch results are empty, check image quality and orientation.
