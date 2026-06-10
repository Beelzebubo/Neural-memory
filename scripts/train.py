import argparse
import json
import logging
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import TextEmbedder, VectorMemoryStore


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("neural_memory.train")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def load_data(path: str) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix == ".jsonl":
        texts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    texts.append(record.get("text", record.get("content", line)))
        if not texts:
            raise ValueError(f"No valid text entries found in {path}")
        return texts

    with open(path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    if not texts:
        raise ValueError(f"No non-empty lines found in {path}")
    return texts


def train_embeddings(
    texts: list[str],
    embedder: TextEmbedder,
    store: VectorMemoryStore,
    logger: logging.Logger,
) -> VectorMemoryStore:
    logger.info(f"Generating embeddings for {len(texts)} texts (dim={len(embedder)})")

    batch_size = 64
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        batch = texts[i : i + batch_size]
        try:
            embeddings = embedder.embed_batch(batch)
        except Exception as e:
            logger.error(f"Failed to embed batch starting at index {i}: {e}")
            continue

        for j, (text, emb) in enumerate(zip(batch, embeddings)):
            entry_id = f"mem_{i + j}"
            store.store(entry_id, emb, {"source": text, "importance_score": 0.5})

    logger.info(f"Stored {len(store)} embeddings in memory store")
    return store


def main():
    parser = argparse.ArgumentParser(description="Train neural memory embeddings")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--data", type=str, required=True, help="Path to training data (txt or jsonl)")
    parser.add_argument("--output", type=str, default="data/memory_store.pkl", help="Path to save trained model")
    args = parser.parse_args()

    logger = setup_logging()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    logger.info("Initializing embedder...")
    try:
        model_name = config.get("embedder", {}).get("model_name", "all-MiniLM-L6-v2")
        embedder = TextEmbedder(model_name=model_name)
    except Exception as e:
        logger.error(f"Failed to initialize embedder: {e}")
        sys.exit(1)

    logger.info("Loading training data...")
    try:
        texts = load_data(args.data)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    store = VectorMemoryStore()
    train_embeddings(texts, embedder, store, logger)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        store.save(str(output_path))
        logger.info(f"Model saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
