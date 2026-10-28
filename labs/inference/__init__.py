"""Running models for people to try: a worker process that loads pretrained and trained models
once and answers requests (generate text, embed, transcribe, classify, caption, detect, predict
with a finished run). No Qt; the app talks to it through ``app/services/inference.py``."""
