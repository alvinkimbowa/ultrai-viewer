# UltrAi Viewer User Guide

Version 0.1.0 (beta). For research use only; results need clinical validation.

## Install

- **Windows:** run `UltrAiNerveViewer-v0.1.0-windows-x64-installer.exe`, then open **UltrAi Nerve Viewer** from the Start menu or the desktop.
- **Linux:** run `UltrAiNerveViewer-v0.1.0-linux-x86_64` in the folder this guide is in.

Runs on the CPU; a 4-core CPU and 8 GB of RAM are recommended. macOS is not supported.

## Load files

Click **Load files** and choose images (PNG, JPEG, TIFF, BMP) or videos, and an output folder.

Scroll, use the arrow keys, or **Prev** / **Next** to move between images and frames. **N** and **P** go to the next and previous file.

## Classes and objects

Each mask is an object, tagged on the image with its class and number, such as **Vessel #1**. Every object of a class has the class's colour.

- **Draw first, name later:** an object you draw without choosing a class is **Unlabeled**, in grey.
- **Name an object:** double-click it under **Objects**, or right-click it there or on the image, and choose a class or type a new one. Renaming numbers it on from that class's objects, such as **Vessel #2**.
- **Give new objects a class:** click the class under **Objects**. Click **Unlabeled** to go back to drawing unnamed objects.
- **Rename a class:** double-click it under **Objects**, or right-click it. The objects on screen take the new name.
- **Remove a class:** right-click it. This is offered only while no object on screen has the class.
- **Show or hide:** untick an object to hide it, or a class to hide all its objects. A hidden object stays hidden on the other frames of the video, is still saved and measured, and cannot be selected or erased.
- **Select:** clicking an object under **Objects** selects it on the image, and selecting it on the image selects it in the list.

The classes are kept per output folder. Classes found on masks saved earlier are added to the list.

## Draw and edit masks

- **Freehand Line** and **Segmented Line** draw an outline. With **Fill mask** ticked, each outline becomes a new object.
- **Paint Brush** paints into the selected mask, or starts a new one. **Eraser** erases from the selected mask only, or from every mask shown when none is selected. A mask stays selected when you change tools; press Esc to deselect it.
- With any tool, click a mask's name tag to select it, and click it again to deselect it. Drag the tag to move the mask. A stroke that starts off a tag paints, erases or draws across tags as usual.
- With **Select**, you can also drag the mask itself to move it. Inside an ROI box, drag off the masks to move the box.
- Right-click a mask, or its name tag with any tool, to name it or to **Delete mask**. You can also click it with **Select** and press Delete.
- **Delete masks** in the sidebar deletes every mask of the image or frame on screen.
- In a video, **Delete before** deletes the masks of every frame up to and including the frame on screen, and **Delete after** deletes those of the frame on screen and every frame after it. With a mask selected, they delete only the masks of its class and number, such as every **Vessel #1**, and leave the rest. Both ask before deleting.
- Ctrl+Z and Ctrl+Y undo and redo.

## Segment with a model

1. Choose a model under **Model**.
2. Click **Segment image** for the image or frame on screen, **Segment video** for the whole video, or **Segment all files** for everything loaded.

A model that segments one structure names its masks with the class chosen under **Objects**, or **Median** when none is chosen. The model's masks replace masks of the same class; masks of other classes stay.

A video model, such as V-JEPA, reads 16 frames at a time, so a frame is segmented from the frames around it. Segmenting one frame reads its neighbours too, and gives the same masks as segmenting the whole video. On a single image, it treats the image as a still video.

## Propagate masks with SAM 3.1

**SAM 3.1** appears under **Model** on a computer set up with Meta's SAM 3 code and an NVIDIA GPU. It spreads the masks of a few frames of a video over the rest of it. On images, the prompt tools and **Find** work the same way; there is nothing to propagate.

1. Choose **SAM 3.1** under **Model**. The SAM controls take the place of the Segment buttons, and **Point prompt**, **Box prompt** and **Mask prompt** are added under **Tools**.
2. Give SAM masks to start from, on as many frames as you like:
   - Masks already drawn or annotated are used as they are.
   - **Point prompt**: left-click to include a spot, right-click to exclude one. **Box prompt**: drag a box around the structure. **Mask prompt**: outline the structure roughly, either by dragging round it or by clicking its corners and then double-clicking or right-clicking to close the outline. SAM fits the mask to the structure, and later points and boxes refine it, while a new outline starts it again. Each prompt updates the mask. Press **New object** before prompting the next structure. Click a point to remove it. Ctrl+Z takes back the last prompt and Ctrl+Y puts it back; Esc clears every prompt and the mask they made. To redo a mask, select it first.
   - **Text prompt** and **Find** add a mask for each match of a phrase.
   - Masks get the class chosen under **Objects**, or are **Unlabeled**; select a mask first to change it instead. **Find** names its matches with the class chosen, or else with the text searched. A mask with the same class and number on several frames (such as **Vessel #1**) is followed as one structure.
3. Go to the frame to start from and click **Propagate video**. **Both directions** fills the frames before and after it. Untick it to choose **Forward** or **Backward**.
   - **Propagate all videos** does every loaded video in turn, each through all its frames from its first prompt frame. Videos without masks are skipped.
   - A video is saved only once it is finished, so Cancel keeps the videos already done and leaves the one in progress as it was. The next Propagate all videos asks whether to propagate the finished videos again; **No** skips them and carries on with the rest. A video whose prompt frames changed since counts as not finished.
4. Check the result, then trim it with **Delete before** / **Delete after**, or correct a frame by hand. The masks you change by hand are used as prompts the next time you propagate; SAM's other masks on that frame are tracked again. Your annotated frames are prompts in full.

SAM's masks go in `<output folder>/sam3.1_predictions/`; your own annotation files are not changed. Selecting SAM 3.1 loads its model in the background, which takes several seconds; a prompt given before it is ready waits for it.

## Prompt masks with UltraSam

**UltraSam** appears under **Model** on a computer set up with the `ultrasam` project. It is SAM trained on ultrasound: it makes one mask at a time from a point or a box, on an image or a video frame, and does not propagate.

1. Choose **UltraSam** under **Model**. **Point prompt**, **Box prompt** and **Mask prompt** are added under **Tools**.
2. **Box prompt**: drag a box around the structure. **Point prompt**: left-click on it. **Mask prompt**: outline it roughly, by dragging or by clicking its corners and double-clicking to close; UltraSam fits the outline to the structure. UltraSam was trained on a single prompt per mask, so it takes one point, one box or one outline at a time. A prompt of another kind, or a second point, replaces the one given once you confirm it; tick **Remember choice** to replace without being asked for the rest of the session. A box or outline replaces the previous box or outline without asking. UltraSam has no exclude points, so a right-click with **Point prompt** is refused. Click a point to remove it. Ctrl+Z takes back the last prompt and Ctrl+Y puts it back; Esc clears every prompt and the mask they made. Press **New object** before prompting the next structure. To redo a mask, select it first.
3. Masks get the class chosen under **Objects**, or are **Unlabeled**.

UltraSam's masks go in `<output folder>/ultrasam_predictions/`. Selecting UltraSam loads its model in the background, which takes a few seconds; a prompt given before it is ready waits for it. After that, prompts take a fraction of a second.

## Prompt masks with nnInteractive

**nnInteractive** appears under **Model** on a computer set up with the `nnInteractive` project. It is MIC-DKFZ's promptable nnU-Net, trained on CT, MRI and other 3D scans rather than ultrasound. Like UltraSam, it makes one mask at a time on an image or a video frame, and does not propagate.

1. Choose **nnInteractive** under **Model**. **Point prompt**, **Box prompt** and **Mask prompt** are added under **Tools**.
2. Prompt as with SAM 3.1: points, a box and an outline combine. With **Box prompt**, drag with the right button for an exclude box, drawn in red, around what the mask must leave out; you can draw several. nnInteractive takes the prompts one at a time in the order you give them, each refining the mask before it, so the latest prompt weighs most. A box or an outline works far better than a point. On breast ultrasound, the mean Dice is about 0.85 from a box, 0.89 from a rough outline, and 0.47 from a single point. Right-click with **Point prompt** to exclude a spot. Press **New object** before prompting the next structure.
3. Masks get the class chosen under **Objects**, or are **Unlabeled**.

nnInteractive's masks go in `<output folder>/nninteractive_predictions/`. Selecting nnInteractive loads its model in the background, which takes several seconds; a prompt given before it is ready waits for it. Its weights are licensed for non-commercial use only (CC BY-NC-SA 4.0).

## Measurements

1. **Calibrate:** enter **Pixels per mm**, or click **Calibrate** and drag the lines onto a known distance. Until then, areas are in pixels.
2. **Read the results** in the **Measurements** panel: the area of each mask on screen.

## Where results are saved

Everything is saved automatically as you go:

- **Masks:** `<output folder>/<model_name>_predictions/<file name>/`, one PNG per mask.
  - Images: `vessel_001.png`; an object not named yet is `unlabeled_001.png`.
  - Videos: `frame_000012_vessel_001.png`.
  - With **No model** selected, masks go in the output folder itself.
- **Measurements:** `<output folder>/measurements_<model_name>.csv`, one row per mask with its class, number and area.
- **Classes:** `<output folder>/classes.json` holds the class list, in the order that gives their colours.

## Tips

- The models are most reliable when they see only the ultrasound field of view and the image is upright:
  - **Auto ROI** boxes the field of view on the current image or on all images. You can also draw a box with **ROI Box** under **Tools**. Boxes can be moved, resized or deleted.
  - **Rotate image** straightens a tilted image. The image file itself is not changed.
