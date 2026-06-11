# Security Audit Report — neural-memory

**Scope:** `ui/` and `integration/` directories  
**Date:** 2026-06-10  
**Severity key:** 🔴 CRITICAL | 🟠 HIGH | 🟡 MEDIUM | ⚪ LOW

---

## 🔴 CRITICAL — Insecure Pickle Deserialization (Code Execution)

**Files:** `src/memory_store.py:115,119` (called from `ui/app.py:62,67`, `integration/hermes_plugin.py:83,88`)

`VectorMemoryStore.save()` uses `pickle.dump()` and `load()` uses `pickle.load()` with no integrity or authenticity verification. `pickle.load()` can execute arbitrary Python code during deserialization. An attacker who can write to the `.pkl` file (or trick the app into loading a crafted path) gains remote code execution.

**Fix:** Replace pickle with a safe format (JSON + manual numpy serialization) or sign/verify pickle blobs with HMAC. Until then, at minimum restrict file permissions on the store file (`0600`).

---

## 🟠 HIGH — Stored/Reflected XSS via Error Message in innerHTML

**File:** `ui/static/js/app.js:427`

```javascript
container.innerHTML = `<div class="text-secondary">Failed to load: ${e.message}</div>`;
```

`e.message` originates from the API error response body (e.g., `err.detail`). Many API endpoints reflect user-controlled data in error messages:

- `ui/app.py:261` — `detail=str(e)` (exception may include user's `text` input)
- `ui/app.py:369` — `detail=f"Invalid JSON: {e}"` (reflects JSON parse errors)

An attacker who sends malicious input triggers an error whose `detail` field contains unescaped HTML/JS → XSS in any user viewing the UI.

**Fix:** Use `escapeHtml(e.message)` on line 427, and sanitize server-side error messages to never reflect raw user input.

---

## 🟠 HIGH — Missing CSRF Protection

**File:** `ui/app.py:20`

The FastAPI app mounts state-changing endpoints (`POST /api/memories`, `PUT /api/memories/{id}`, `DELETE /api/memories/{id}`, `POST /api/restore`) with no CSRF tokens, double-submit cookies, or `SameSite` enforcement. While JSON Content-Type mitigates simple form-based CSRF, it's not a complete defense (CORS misconfiguration or browser quirks can bypass).

**Fix:** Add CSRF middleware (e.g., `starlette-csrf`) or set `SameSite=Lax`/`Strict` on session cookies.

---

## 🟡 MEDIUM — Missing Security Headers

**File:** `ui/app.py:20`

The FastAPI app sets no security-related HTTP headers:

| Header | Risk |
|--------|------|
| `Content-Security-Policy` | No CSP allows inline scripts (XSS amplification) |
| `X-Content-Type-Options: nosniff` | MIME-sniffing could downgrade security |
| `X-Frame-Options: DENY` | Clickjacking of the UI |
| `Referrer-Policy` | Referrer leakage |

**Fix:** Add a FastAPI middleware that sets security headers.

---

## 🟡 MEDIUM — Arbitrary Keyword Arguments in Plugin execute()

**File:** `integration/hermes_plugin.py:131`

```python
handler(**arguments)
```

`arguments` is user-controlled dictionary passed from `execute(tool_name, arguments)`. If any `cmd_*` method accepts `**kwargs` or has parameters with unsafe defaults, an attacker can inject unexpected arguments. This is a form of parameter injection/prototype pollution.

**Fix:** Validate that `arguments` only contains keys matching the target method's signature. Wrap in a whitelist check.

---

## 🟡 MEDIUM — Unvalidated Embedding Vectors in /api/restore

**File:** `ui/app.py:375-380`

```python
embedding = mem.get("embedding")
if embedding is not None:
    memory_system.store.store(mid, embedding, metadata)
```

No validation on:
- Embedding dimension (does it match store dimension?)
- Embedding type (is it a list of floats?)
- Embedding value range (NaN/Inf can corrupt index)

Malformed embeddings can corrupt the vector index or crash the server.

**Fix:** Validate type, dimension, and numeric sanity before storing.

---

## 🟡 MEDIUM — Information Disclosure via /api/config

**File:** `ui/app.py:341-343`

```python
@app.get("/api/config")
async def get_config():
    return memory_system.config
```

Returns the full configuration dictionary unredacted. May leak:
- Internal file paths
- Model names/versions
- Storage paths
- Operational parameters

**Fix:** Redact sensitive fields or require authentication.

---

## 🟡 MEDIUM — Information Disclosure via /api/stats

**File:** `ui/app.py:309-324`

Exposes `store_path` (the full filesystem path to the pickle file) to any client. Combined with the pickle deserialization issue, this tells an attacker exactly where to write a malicious file.

**Fix:** Redact or hash the path in the response.

---

## ⚪ LOW — CLI Accepts Arbitrary Store Path (Combined with Pickle = Code Execution)

**Files:** `integration/cli.py:16`, `integration/hermes_plugin.py:56`

```python
parser.add_argument("--store-path", help="Path to the memory store pickle file")
...
self.store_path = Path(store_path or ...)
```

An attacker who can invoke the CLI with a crafted `--store-path` pointing to a malicious pickle achieves code execution. Even without pickle, reading arbitrary files via path traversal is possible.

**Fix:** Validate the store path is within an allowed directory, or restrict to default location.

---

## ⚪ LOW — No Input Size Limits (DoS Risk)

**File:** `ui/app.py:250-261`

The `POST /api/memories` endpoint accepts arbitrary-length text (no max length), unbounded tag arrays, and unlimited importance values. An attacker could:
- Upload multi-MB text blocks (memory exhaustion)
- Send thousands of tag entries
- Overwhelm the embedding model

**Fix:** Add `max_length` to text, `max_items` to tags, and validate importance range.

---

## ⚪ LOW — Embedding Data Returned to Client

**Files:** `ui/app.py:75,93-96,354-357`

Embedding vectors (high-dimensional numeric representations of user data) are returned to the client in `GET /api/memories`, `GET /api/memories/{id}`, and `POST /api/backup`. Embeddings can be used for model inversion or membership inference attacks.

**Fix:** Make embedding return opt-in (e.g., `?include_embeddings=true`) rather than always-on.
