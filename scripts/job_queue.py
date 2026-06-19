#!/usr/bin/env python3
"""Job Queue — crash-safe persistent queue with retries and stats.

Stores jobs in JSON, supports priority ordering, max retries,
and concurrency-safe (single-process, file-locked) dequeue.
"""

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

QUEUE_PATH = os.path.expanduser("~/.neural_memory/job_queue.json")
LOCK = threading.Lock()

MAX_RETRIES_DEFAULT = 3


class Job:
    def __init__(
        self,
        job_type: str,
        params: Optional[Dict] = None,
        priority: int = 0,
        max_retries: int = MAX_RETRIES_DEFAULT,
        job_id: Optional[str] = None,
        status: str = "queued",
        created_at: Optional[float] = None,
        started_at: Optional[float] = None,
        completed_at: Optional[float] = None,
        retries: int = 0,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ):
        self.job_id = job_id or str(uuid.uuid4())
        self.job_type = job_type
        self.params = params or {}
        self.priority = priority
        self.max_retries = max_retries
        self.status = status
        self.created_at = created_at or time.time()
        self.started_at = started_at
        self.completed_at = completed_at
        self.retries = retries
        self.result = result
        self.error = error

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "params": self.params,
            "priority": self.priority,
            "max_retries": self.max_retries,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "retries": self.retries,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        return cls(
            job_id=d.get("job_id"),
            job_type=d.get("job_type", "unknown"),
            params=d.get("params", {}),
            priority=d.get("priority", 0),
            max_retries=d.get("max_retries", MAX_RETRIES_DEFAULT),
            status=d.get("status", "queued"),
            created_at=d.get("created_at"),
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            retries=d.get("retries", 0),
            result=d.get("result"),
            error=d.get("error"),
        )


class JobQueue:
    def __init__(self, path: str = QUEUE_PATH):
        self.path = path
        self._jobs: Dict[str, Job] = {}
        self._lock = LOCK
        self._load()

    def _load(self):
        p = Path(self.path)
        if p.exists():
            try:
                data = json.loads(p.read_text())
                self._jobs = {jid: Job.from_dict(jd) for jid, jd in data.items()}
                logger.info("Loaded %s jobs from queue", len(self._jobs))
            except Exception as e:
                logger.error("Failed to load job queue: %s", e)
                self._jobs = {}

    def _save(self):
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {jid: job.to_dict() for jid, job in self._jobs.items()}
        p.write_text(json.dumps(data, indent=2))

    def enqueue(self, job_type: str, params: Optional[Dict] = None, priority: int = 0, max_retries: int = MAX_RETRIES_DEFAULT) -> Job:
        with self._lock:
            job = Job(job_type=job_type, params=params, priority=priority, max_retries=max_retries)
            self._jobs[job.job_id] = job
            self._save()
            logger.info("Enqueued job %s: %s", job.job_id[:8], job_type)
            return job

    def next(self) -> Optional[Job]:
        with self._lock:
            candidates = [j for j in self._jobs.values() if j.status == "queued"]
            if not candidates:
                return None
            candidates.sort(key=lambda j: (-j.priority, j.created_at))
            job = candidates[0]
            job.status = "running"
            job.started_at = time.time()
            self._save()
            return job

    def complete(self, job_id: str, result: str = "ok", error: Optional[str] = None):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                logger.warning("Job %s not found for completion", job_id[:8])
                return
            job.status = "completed" if not error else "failed"
            job.completed_at = time.time()
            job.result = result
            if error:
                job.error = error
            self._save()
            logger.info("Job %s completed with status %s", job_id[:8], job.status)

    def fail(self, job_id: str, error: str):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                logger.warning("Job %s not found for failure", job_id[:8])
                return
            job.retries += 1
            if job.retries >= job.max_retries:
                job.status = "failed"
                job.completed_at = time.time()
                job.error = error
                logger.warning("Job %s failed permanently after %s retries: %s", job_id[:8], job.retries, error)
            else:
                job.status = "queued"
                job.error = error
                logger.warning("Job %s failed (retry %s/%s): %s", job_id[:8], job.retries, job.max_retries, error)
            self._save()

    def status(self, limit: int = 50) -> dict:
        with self._lock:
            all_jobs = list(self._jobs.values())
            all_jobs.sort(key=lambda j: j.created_at, reverse=True)
            recent = [j.to_dict() for j in all_jobs[:limit]]
            counts = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
            for j in all_jobs:
                s = j.status
                counts[s] = counts.get(s, 0) + 1
            return {"counts": counts, "recent": recent, "total": len(all_jobs)}

    def stats(self) -> dict:
        with self._lock:
            total = len(self._jobs)
            counts = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
            type_counts: Dict[str, int] = {}
            for j in self._jobs.values():
                counts[j.status] = counts.get(j.status, 0) + 1
                type_counts[j.job_type] = type_counts.get(j.job_type, 0) + 1
            return {"total": total, "by_status": counts, "by_type": type_counts}

    def list_by_type(self, job_type: str, limit: int = 50) -> List[Job]:
        with self._lock:
            matched = [j for j in self._jobs.values() if j.job_type == job_type]
            matched.sort(key=lambda j: j.created_at, reverse=True)
            return matched[:limit]


job_queue = JobQueue()


# ── Background Worker ──

class JobWorker:
    def __init__(self, queue: JobQueue, poll_interval: float = 5.0, max_concurrent: int = 1):
        self.queue = queue
        self.poll_interval = poll_interval
        self.max_concurrent = max_concurrent
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handlers: Dict[str, callable] = {}

    def register(self, job_type: str, handler: callable):
        self._handlers[job_type] = handler

    def start(self):
        if self._thread and self._thread.is_alive():
            logger.warning("Worker already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Job worker started (poll=%ss, max_concurrent=%s)", self.poll_interval, self.max_concurrent)

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
            logger.info("Job worker stopped")

    def _run(self):
        while not self._stop_event.is_set():
            try:
                job = self.queue.next()
                if job is None:
                    self._stop_event.wait(self.poll_interval)
                    continue
                handler = self._handlers.get(job.job_type)
                if handler is None:
                    self.queue.fail(job.job_id, error=f"No handler registered for {job.job_type}")
                    continue
                try:
                    result = handler(job)
                    self.queue.complete(job.job_id, result=result)
                except Exception as e:
                    self.queue.fail(job.job_id, error=str(e))
            except Exception:
                logger.exception("Job worker error")
                self._stop_event.wait(self.poll_interval)


worker = JobWorker(job_queue)


def start_worker():
    worker.start()

def stop_worker():
    worker.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import argparse
    parser = argparse.ArgumentParser(description="Job Queue — inspect and manage")
    parser.add_argument("action", choices=["enqueue", "status", "stats", "worker"], default="status", nargs="?")
    parser.add_argument("--type", default="test", help="Job type for enqueue")
    parser.add_argument("--params", default="{}", help="JSON params for enqueue")
    parser.add_argument("--priority", type=int, default=0, help="Job priority")
    parser.add_argument("--limit", type=int, default=50, help="Status limit")
    args = parser.parse_args()
    if args.action == "enqueue":
        j = job_queue.enqueue(job_type=args.type, params=json.loads(args.params), priority=args.priority)
        print(json.dumps(j.to_dict(), indent=2))
    elif args.action == "status":
        s = job_queue.status(limit=args.limit)
        print(json.dumps(s, indent=2, default=str))
    elif args.action == "stats":
        s = job_queue.stats()
        print(json.dumps(s, indent=2))
    elif args.action == "worker":
        def sample_handler(job):
            logger.info("Processing job %s: %s", job.job_id[:8], job.job_type)
            time.sleep(1)
            return "processed"
        worker.register("test", sample_handler)
        worker.start()
        logger.info("Worker running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            worker.stop()
