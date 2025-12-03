# HTTPx Timeout Configuration

For long-running API calls (Deep Research, batch processing, etc.), configure explicit timeouts.

## The Problem

Default httpx timeout is ~10 minutes for reads. Deep Research API calls can take 15-30+ minutes.

## Solution

Use `httpx.Timeout` with granular control:

```python
import httpx
from openai import AsyncOpenAI

# Deep Research can take 15-30+ minutes
LONG_RUNNING_TIMEOUT = httpx.Timeout(
    connect=30.0,    # 30s to establish connection
    read=1800.0,     # 30 min to wait for response (key setting)
    write=60.0,      # 60s to send request
    pool=30.0,       # 30s to acquire connection from pool
)

client = AsyncOpenAI(
    api_key=api_key,
    timeout=LONG_RUNNING_TIMEOUT,
)
```

## Timeout Components

- **connect**: Time to establish TCP connection
- **read**: Time to wait for response data (this is the one that usually needs increasing)
- **write**: Time to send request body
- **pool**: Time to acquire connection from pool

## Rule of Thumb

- Standard API calls: Default is fine
- Batch/async jobs: 5-10 minutes read timeout
- Deep Research: 30+ minutes read timeout
- Always log when starting long calls so users know it's working

## Logging for Long Calls

```python
logger.info("🌐 Starting deep research (this may take several minutes)...")
start = time.time()
response = await client.responses.create(...)
logger.info(f"✅ Completed in {time.time() - start:.1f}s")
```

