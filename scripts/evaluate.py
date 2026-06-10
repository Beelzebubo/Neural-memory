import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import TextEmbedder, VectorMemoryStore, MemoryRetriever


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("neural_memory.evaluate")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def load_test_data(path: str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test data file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            records = [json.loads(line) for line in f if line.strip()]
        else:
            records = json.load(f)

    if isinstance(records, dict):
        records = [records]

    validated = []
    for r in records:
        if "query" not in r:
            raise ValueError(f"Each test record must contain a 'query' field: {r}")
        relevant = r.get("relevant", r.get("relevant_ids", []))
        if isinstance(relevant, str):
            relevant = [relevant]
        validated.append({"query": r["query"], "relevant_ids": list(relevant)})

    if not validated:
        raise ValueError("No valid test queries found")

    return validated


def recall_at_k(retrieved: list[dict], relevant_ids: set, k: int) -> float:
    top_k = [r["id"] for r in retrieved[:k]]
    if not relevant_ids:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def average_precision(retrieved: list[dict], relevant_ids: set) -> float:
    if not relevant_ids:
        return 0.0
    hits = 0
    sum_precision = 0.0
    for i, r in enumerate(retrieved):
        if r["id"] in relevant_ids:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / len(relevant_ids) if relevant_ids else 0.0


def mean_reciprocal_rank(retrieved: list[dict], relevant_ids: set) -> float:
    for i, r in enumerate(retrieved):
        if r["id"] in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def print_metrics_table(results: list[dict], k_values: list[int]):
    header = f"{'Query':<40} " + "".join(f"{'R@{:<5}':>8}" for k in k_values) + f"{'MAP':>8} {'MRR':>8}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for r in results:
        row = f"{r['query'][:39]:<40} "
        for k in k_values:
            row += f"{r[f'recall@{k}']:>8.3f}"
        row += f"{r['map']:>8.3f} {r['mrr']:>8.3f}"
        print(row)

    if len(results) > 1:
        print(sep)
        avg = {
            "query": "AVERAGE",
            **{f"recall@{k}": sum(r[f"recall@{k}"] for r in results) / len(results) for k in k_values},
            "map": sum(r["map"] for r in results) / len(results),
            "mrr": sum(r["mrr"] for r in results) / len(results),
        }
        row = f"{avg['query']:<40} "
        for k in k_values:
            row += f"{avg[f'recall@{k}']:>8.3f}"
        row += f"{avg['map']:>8.3f} {avg['mrr']:>8.3f}"
        print(row)
        print(sep)

    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate neural memory retrieval accuracy")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test data (json or jsonl)")
    parser.add_argument("--output", type=str, default=None, help="Path to save evaluation results (optional)")
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

    persistence_path = config.get("store", {}).get("persistence", {}).get("save_path", "data/memory_store.pkl")
    project_root = Path(__file__).resolve().parent.parent
    persistence_path = str(project_root / persistence_path)

    logger.info(f"Loading memory store from {persistence_path}")
    store = VectorMemoryStore()
    try:
        store.load(persistence_path)
    except FileNotFoundError:
        logger.error(f"No saved memory store found at {persistence_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load memory store: {e}")
        sys.exit(1)

    if len(store) == 0:
        logger.warning("Memory store is empty — evaluation will produce zero scores")

    logger.info(f"Loaded memory store with {len(store)} entries (dim={store._dim})")

    logger.info("Initializing embedder...")
    try:
        model_name = config.get("embedder", {}).get("model_name", "all-MiniLM-L6-v2")
        embedder = TextEmbedder(model_name=model_name)
    except Exception as e:
        logger.error(f"Failed to initialize embedder: {e}")
        sys.exit(1)

    retriever = MemoryRetriever(embedder, store)
    default_k = config.get("retriever", {}).get("default_k", 10)
    k_values = [1, 3, 5, default_k]

    logger.info(f"Loading test data from {args.test_data}")
    try:
        test_queries = load_test_data(args.test_data)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    if not test_queries:
        logger.error("Test data is empty")
        sys.exit(1)

    logger.info(f"Running evaluation on {len(test_queries)} queries...")
    results = []
    for tq in test_queries:
        try:
            retrieved = retriever.retrieve(tq["query"], k=max(k_values))
        except Exception as e:
            logger.error(f"Failed to retrieve for query '{tq['query']}': {e}")
            continue

        relevant_ids = set(tq["relevant_ids"])
        row = {"query": tq["query"]}
        for k in k_values:
            row[f"recall@{k}"] = recall_at_k(retrieved, relevant_ids, k)
        row["map"] = average_precision(retrieved, relevant_ids)
        row["mrr"] = mean_reciprocal_rank(retrieved, relevant_ids)
        results.append(row)

    if not results:
        logger.error("No queries could be evaluated")
        sys.exit(1)

    print_metrics_table(results, k_values)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    total_map = sum(r["map"] for r in results) / len(results)
    total_mrr = sum(r["mrr"] for r in results) / len(results)
    logger.info(f"Evaluation complete — MAP: {total_map:.4f}, MRR: {total_mrr:.4f}")


if __name__ == "__main__":
    main()
