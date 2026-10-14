# UltAI Viewer User Guide

Version 1.1.0 (beta). For research use only; results need clinical validation.

## Install

Download the app from the [UltrAi Viewer folder](https://ubcca-my.sharepoint.com/:f:/r/personal/alvinbk_student_ubc_ca/Documents/UltrAi%20Research/Musculoskeletal%20ultrasound/Ultrasound%20for%20Osteoarthritis/Desktop%20App/UltrAi%20Viewer?csf=1&web=1&e=vF2m8l).

- **Windows:** run `UltrAiViewer-v1.1.0-windows-x64-installer.exe`, then open **UltrAi Viewer** from the Start menu or the desktop.
- **Linux:** extract `UltrAiViewer-v1.1.0-linux-x86_64.tar.gz` and run `UltrAiViewer-v1.1.0-linux-x86_64` inside the extracted folder.

Runs on the CPU; a 4-core CPU and 8 GB of RAM are recommended. macOS is not supported.

## Load and segment

1. Click **Load files** and choose images (PNG, JPEG, TIFF, BMP) and an output folder.
2. Choose a model under **Model** (nnunet recommended for highest precision).
3. Click **Segment image** for the image on screen, or **Segment all files** for all loaded images.
4. Correct the mask if needed with the tools under **Tools**. Ctrl+Z and Ctrl+Y undo and redo.

Scroll, use the arrow keys, or **Prev** / **Next** to move between images.

Masks are saved automatically as you go to `<output folder>/<model_name>_predictions/<image name>.png`.

## Measurements

1. **Calibrate:** enter **Pixels per mm**, or click **Calibrate** and drag the lines onto a known distance. Until then the default of 1.00 is used, so the results are not real mm.
2. **View:** choose **Full cartilage** or **Per region** (lateral, notch, medial).
3. **Knee:** the side is read from the file name (`_L_`, `_R_`, `_left_`, `_right_`); if the name does not say, choose it.
4. **Notch centre:** if needed, drag the yellow diamond with the **Select** tool.
5. **Read the results** in the **Measurements** panel: area, length, thickness and echo intensity of the cartilage.

Measurements are saved automatically as you go to `<output folder>/measurements_<model_name>.csv`, one row per image.

## Tips

- The models are most reliable when they see only the ultrasound field of view and the image is upright:
  - **Auto ROI** boxes the field of view on the current image or on all images. You can also draw a box with **ROI Box** under **Tools**. Boxes can be moved, resized or deleted.
  - **Rotate image** straightens a tilted image. The image file itself is not changed.

![UltAI Viewer with an ROI, a segmented image and its measurements per region](docs/images/app_window.png)
