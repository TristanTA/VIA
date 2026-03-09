# VIA

Vision-assisted information assistant: a small pipeline that labels objects/scenes in an image and produces filtered outputs.

## Repo layout
- `main.py`: entrypoint
- `image_processing.py`: image pre-processing/helpers
- `models/`: YOLOv8 wrapper
- `outputs/`: example generated outputs

## Quickstart
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt  # (add if missing)
python main.py
```

## Notes
- This repo currently includes large / generated artifacts (e.g. model weights, crops, temporary images). For a cleaner portfolio repo, consider moving them to releases or adding them to `.gitignore`.
