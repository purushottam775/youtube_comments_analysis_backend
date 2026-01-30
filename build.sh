#!/usr/bin/env bash
set -o errexit

pip install -r requirements.txt

# Pre-download the model so the first request doesn't timeout
python -c "from transformers import pipeline; pipeline('text-classification', model='nlptown/bert-base-multilingual-uncased-sentiment')"
