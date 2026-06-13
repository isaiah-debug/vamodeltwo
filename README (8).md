{
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": [
    "# 01 — Explore a transcript\n",
    "Run the audio stage first (see README Quickstart). Logic lives in `src/`; this notebook only calls it."
  ]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": [
    "import json, sys\n",
    "from pathlib import Path\n",
    "sys.path.insert(0, str(Path.cwd().parent / 'src'))\n",
    "from pipeline.features.turns import compute_speaker_features\n",
    "\n",
    "SESSION = 's01_groupA'  # <- change me\n",
    "transcript = json.load(open(next((Path.cwd().parent / 'output' / SESSION).glob('*.json'))))\n",
    "compute_speaker_features(transcript)"
  ]}
 ],
 "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
              "language_info": {"name": "python"}},
 "nbformat": 4, "nbformat_minor": 5
}
