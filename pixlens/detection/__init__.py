import logging

logging.basicConfig(
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(module)-30s | %(message)s",
    handlers=[logging.StreamHandler()],
)
