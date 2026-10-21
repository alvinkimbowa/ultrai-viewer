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

Scroll, use the arrow keys, or **Prev** / **Next** to move between images and frames. **N** and **P** go to the next and previous file.

## Draw and edit masks

- **Freehand Line** and **Segmented Line** draw an outline. With **Fill mask** ticked, each outline becomes a new mask of the picked label.
- **Paint Brush** paints into the selected mask, or starts a new one. **Eraser** erases from the selected mask only, or from every mask shown when none is selected. A mask stays selected when you change tools; press Esc to deselect it.
- With **Paint Brush** or **Eraser**, click a mask's name tag to select it, and click it again to deselect it. A stroke that starts off a tag paints or erases across tags as usual.
- **Masks shown**, at the end of **Edit**, lists the masks on screen. Untick one to hide it, or use **Select all** to show or hide them all. A hidden mask stays hidden on the other frames of the video, is still saved and measured, and cannot be selected or erased.
- With **Select**, drag a mask to move it. Inside an ROI box, drag off the masks to move the box.
- Each mask is tagged with its label and number, such as **Median #1**.
- Right-click a mask to change its **Label** or to **Delete mask**. You can also click it with **Select** and press Delete.
- **Delete masks** in the sidebar deletes every mask of the image or frame on screen.
- In a video, **Delete before** deletes the masks of every frame up to and including the frame on screen, and **Delete after** deletes those of the frame on screen and every frame after it. With a mask selected, they delete only the masks of its label and number, such as every **Median #1**, and leave the rest. Both ask before deleting.
- Ctrl+Z and Ctrl+Y undo and redo.

## Segment with a model

1. Choose a model under **Model**.
2. Click **Segment image** for the image or frame on screen, **Segment video** for the whole video, or **Segment all files** for everything loaded.

A model that segments one structure labels its masks with the nerve picked, or **Median** when none is picked. The model's masks replace masks of the same label; masks of other labels stay.

## Propagate masks with SAM 3.1

**SAM 3.1** appears under **Model** on a computer set up with Meta's SAM 3 code and an NVIDIA GPU. It spreads the masks of a few frames of a video over the rest of it.

1. Choose **SAM 3.1** under **Model**. The SAM controls take the place of the Segment buttons, and **SAM Point** and **SAM Box** are added under **Tools**.
2. Give SAM masks to start from, on as many frames as you like:
   - Masks already drawn or annotated are used as they are.
   - **SAM Point**: left-click to include a spot, right-click to exclude one. **SAM Box**: drag a box around the structure. Each click updates the mask. Press **New object** before prompting the next structure. To redo a mask, select it first.
   - **Text prompt** and **Find** add a mask for each match of a phrase.
   - Masks get the label picked under **Labels**; pick one before drawing or prompting, or select the mask to change. A mask with the same label and number on several frames (such as **Median #1**) is followed as one structure.
3. Go to the frame to start from and click **Propagate video**. **Both directions** fills the frames before and after it. Untick it to choose **Forward** or **Backward**.
   - **Propagate all videos** does every loaded video in turn, each through all its frames from its first prompt frame. Videos without masks are skipped.
   - A video is saved only once it is finished, so Cancel keeps the videos already done and leaves the one in progress as it was. The next Propagate all videos asks whether to propagate the finished videos again; **No** skips them and carries on with the rest. A video whose prompt frames changed since counts as not finished.
4. Check the result, then trim it with **Delete before** / **Delete after**, or correct a frame by hand. The masks you change by hand are used as prompts the next time you propagate; SAM's other masks on that frame are tracked again. Your annotated frames are prompts in full.

SAM's masks go in `<output folder>/sam3.1_predictions/`; your own annotation files are not changed. The first prompt or propagation loads the model, which takes about half a minute.

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
