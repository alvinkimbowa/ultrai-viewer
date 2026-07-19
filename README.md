# UltrAi Annotator

The existing PyQt desktop application remains in `src/` as a reference. The local, Chrome-only annotation application is in `web/` and does not include the AI segmentation models.

## Launch the web application

- Linux: run `./start_web.sh`
- macOS: double-click `start_web.command`
- Windows: double-click `start_web.bat`

Google Chrome opens the application directly from this folder. No server, internet connection, Python environment, or installation is required.

Choose an output folder before saving masks. Image masks are stored in a folder named after the image as `class_001.png`. Video masks are stored in a folder named after the video as `frame_000000_class_001.png`. The manifest is stored as `nerve_manifest.json` in the selected output folder.
