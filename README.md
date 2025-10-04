# Byte-builders_sentinel

## Project Overview

This project implements a sentinel system for automated judgement and monitoring.

## Automated Judgement Criteria

1. **Design & Implementation Quality** – design, structure, documentation, tests, and overall code hygiene across `src/`.
2. **Accuracy of the Results** – JSON outputs in `evidence/output/` must match the organiser ground truth.
3. **Algorithms Used** – automation searches for `# @algorithm Name | Purpose` markers and inspects the tagged implementations.

## In-person Judgement (2‑minute walk-through. Strictly time boxed)

4. **Quality of the Dashboard** – judges review the clarity and usefulness of your visualisations.
5. **Solution Presentation** – teams present the system within the timebox and address questions.

## Directory Structure

```
Byte-builders_sentinel/
├── README.md              # this file (submission guidelines)
├── SUBMISSION_GUIDE.md    # fill in before submitting
├── src/                   # your complete source code goes here
├── evidence/
│   ├── screenshots/       # dashboard captures (PNG recommended)
│   ├── output/
│   │   ├── test/                # your events.jsonl for test data
│   │   └── final/               # your events.jsonl for final data
│   └── executables/       # automation script + required binaries
```

## Algorithm Tagging

Any function that implements an algorithm must include a line comment immediately above it in the form:

```python
# @algorithm Name | Purpose
def algorithm_function():
    pass
```

The automation only recognises this exact prefix; omitting it will force the "Algorithms" score to 0. If you are working in another language, adapt the comment syntax but keep the `@algorithm Name | Purpose` text verbatim.

## Evidence Expectations

- Place each generated `events.jsonl` in the matching split under `evidence/output/` (`test/` and `final/`) and keep the filenames exactly as provided.
- Keep your executables in `evidence/executables/`, including a single entry point `run_demo.py` that installs dependencies, starts required services, and regenerates the `./output/` tree to mirror `evidence/output/`.
- Note any special prerequisites (network, environment variables, warm-up time) in this README, and remember judges will only execute the single command you document in the SUBMISSION_GUIDE.md—make sure it's crystal clear and self sufficient.
- Capture system visuals in `evidence/screenshots/` using clear, descriptive PNGs (e.g., `dashboard-overview.png`).

## Prerequisites

- Python 3.8+
- Required dependencies (see requirements.txt)
- Network access (if applicable)
- Environment variables (if any)

## Setup Instructions

1. Install dependencies: `pip install -r requirements.txt`
2. Configure environment variables (if any)
3. Run the demo: `python evidence/executables/run_demo.py`

## Packaging Checklist

- [x] Replace `Team##` in directory names with your assigned team number.
- [ ] Verify `src/` contains the final source tree and comment-tagged algorithms (run `grep -R "@algorithm" src`).
- [ ] Regenerate evidence so each `evidence/output/*/events.jsonl` reflects your latest code.
- [ ] Confirm the automation script regenerates the outputs end-to-end.
- [ ] Fill out `SUBMISSION_GUIDE.md` before zipping your final archive.
- [ ] Zip the `Byte-builders_sentinel/` folder directly (no extra nesting) for upload.