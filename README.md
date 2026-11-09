# Eye Tracking Mouse

A webcam-based desktop mouse controller written in Python. It uses MediaPipe Face Mesh iris landmarks and OpenCV to estimate gaze direction, then maps that estimate to screen coordinates after a nine-point calibration.

> **Educational prototype:** this is not a clinical or precision eye tracker. Accuracy depends on lighting, camera placement, head movement, glasses and individual anatomy. Keep a physical mouse or trackpad available while testing.

## Features

- Live face and iris landmark detection through a webcam
- Nine-point calibration with a least-squares mapping
- Smoothed gaze-based pointer movement
- Optional bilateral blink-to-left-click
- Optional long-blink-to-right-click
- Optional dwell-to-left-click (off by default)
- Tkinter control panel with Start, Stop, calibration and settings
- Local settings and calibration persistence
- Local-only video processing; webcam frames are not recorded or uploaded

## Requirements

- Python 3.10 or 3.11 (Python 3.11 recommended)
- Webcam and desktop session
- macOS, Windows or Linux

## Installation

### macOS

    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python app.py

When prompted, allow camera access for Terminal or the IDE under System Settings > Privacy & Security > Camera. To move the pointer and send clicks, enable that same app under System Settings > Privacy & Security > Accessibility. Restart the terminal or IDE after changing permissions.

### Windows

    py -3.11 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python app.py

### Linux

Install Tkinter and the system GUI libraries for your distribution (for Debian/Ubuntu, the package is commonly named python3-tk), then run:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python app.py

## How to use

1. Run the application with python app.py.
2. Select the webcam index (usually 0) and click **Start tracking**.
3. Sit in a well-lit place, face the webcam and keep the camera stable.
4. Click **Calibrate (9 points)**.
5. Look at each displayed screen coordinate in sequence. Keep your head still and click **Capture** after your gaze settles on each target.
6. After calibration, gaze movement controls the pointer. Enable optional gestures in the panel as required.
7. Click **Stop tracking** to immediately stop pointer control and release the camera.

## Mouse gestures

- **Blink = left click:** a short bilateral eye closure triggers a left click after a cooldown.
- **Long blink = right click:** an optional longer bilateral closure triggers a right click. This setting is off by default.
- **Dwell = left click:** keeping the gaze estimate within a small area for the selected duration triggers a left click. This setting is off by default to avoid accidental clicks.

Natural blinking can be mistaken for a click. Disable blink clicks if they are inconvenient. Dwell and blink thresholds can be adjusted in settings.py.

## Local settings

The application stores settings in ~/.eye_tracking_mouse.json and calibration coefficients in ~/.eye_tracking_mouse_calibration.json. These files remain on your computer and are not part of the repository.

Reset calibration and repeat the nine-point process after changing your camera position, monitor setup, seating position or lighting substantially.

## Troubleshooting

- **Camera will not open:** close other applications using the webcam and try camera index 1 or 2.
- **Face or iris not detected:** improve front lighting, move your face fully into frame and reduce glare.
- **Pointer does not move:** complete calibration and check operating-system Accessibility permissions.
- **Pointer jitters:** lower the smoothing slider. Higher values respond faster but may feel less stable.
- **Unwanted clicks:** disable dwell and blink clicks or increase the cooldown in settings.py.
- **MediaPipe installation fails:** use 64-bit Python 3.10 or 3.11 in a fresh virtual environment.

## Project structure

    Eye-Tracking-Mouse/
    ├── app.py             # Tkinter GUI and cursor controller
    ├── eye_tracker.py     # Webcam, Face Mesh and iris feature extraction
    ├── calibration.py     # Nine-point calibration and gaze mapping
    ├── gestures.py        # Blink and dwell gesture state machine
    ├── settings.py        # Local settings persistence
    ├── requirements.txt
    ├── LICENSE
    └── .gitignore

## Safety and privacy

Keep a physical mouse or trackpad within reach. Stop tracking if pointer movement becomes unexpected. Avoid using this prototype for safety-critical tasks. Take breaks and do not intentionally hold your eyes closed for long periods. Camera frames are processed locally and are not uploaded by this application.

## License

This project is released under the MIT License. See LICENSE.
