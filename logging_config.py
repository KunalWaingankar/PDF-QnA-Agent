#Adds the initial log setup required
# src/logging_config.py
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("pdf_qna_log.log"),
            logging.StreamHandler()
        ]
    )
