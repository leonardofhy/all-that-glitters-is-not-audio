"""Shared Claude API utilities for relaxed MCQ evaluation.

Uses Claude Haiku as an MCQ answer extractor: reads model predictions
and determines which option (A/B/C/D) was selected. This replaces
string-match (MMAU/MMAR) and NVEmbed (MMAU-Pro CE) for a more robust
evaluation of verbose/reasoning model outputs.
"""

import asyncio
import os
import re
import time
from pathlib import Path

# Auto-load .env from project root
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
CONCURRENCY = 50  # Simultaneous API requests (Haiku ~4000 RPM)

MCQ_EXTRACTION_PROMPT = """\
Given a model's response to a multiple choice question, determine which answer option the model selected.

Question: {question}

Options:
{options_text}

Model's response:
{prediction}

Which option did the model select? Reply with ONLY the letter ({valid_letters}). \
If the model did not clearly select any option, reply with "NONE"."""


def remove_thinking_process(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Handle leaked/truncated <think> without closing tag (endpoint cutoff)
    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    # Handle </think> without opening <think> (vLLM strips opening tag for some models)
    cleaned = re.sub(r'^.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _build_prompt(question: str, choices: list[str], prediction: str) -> str:
    """Build the MCQ extraction prompt."""
    letters = [chr(65 + i) for i in range(len(choices))]
    options_text = "\n".join(f"{l}) {c}" for l, c in zip(letters, choices))
    valid_letters = ", ".join(letters)
    return MCQ_EXTRACTION_PROMPT.format(
        question=question,
        options_text=options_text,
        prediction=prediction[:2000],  # Truncate overly long predictions
        valid_letters=valid_letters,
    )


def _parse_claude_answer(response_text: str, choices: list[str]) -> str | None:
    """Parse Claude's response into a valid letter or None."""
    letters = [chr(65 + i) for i in range(len(choices))]
    answer = response_text.strip().upper()
    if answer in letters:
        return answer
    if answer == "NONE":
        return None
    # Try extracting first valid letter
    for char in answer:
        if char in letters:
            return char
    return None


def _quick_regex_extract(prediction: str, choices: list[str]) -> str | None:
    """Try to extract answer with regex for obvious single-letter responses.

    Returns the letter if confidently extractable, None otherwise (fall through to Claude).
    """
    prediction = prediction.strip()
    if not prediction:
        return None

    letters = [chr(65 + i) for i in range(len(choices))]

    # Pattern 1: Single letter only (e.g. "B" or "B.")
    m = re.match(r'^([A-Z])\)?\.?\s*$', prediction, re.IGNORECASE)
    if m and m.group(1).upper() in letters:
        return m.group(1).upper()

    # Pattern 2: "The answer is X" pattern
    m = re.search(r'(?:the answer is|answer:\s*|I (?:would )?(?:choose|select|pick))\s*\(?([A-Z])\)?', prediction, re.IGNORECASE)
    if m and m.group(1).upper() in letters:
        return m.group(1).upper()

    return None


async def extract_answers_batch(
    samples: list[dict],
    question_key: str,
    choices_key: str,
    prediction_key: str,
    use_regex_prefilter: bool = True,
) -> tuple[list[str | None], dict]:
    """Batch-extract MCQ answers using Claude Haiku with rate limiting.

    Args:
        samples: List of sample dicts.
        question_key: Key for the question text.
        choices_key: Key for the choices list.
        prediction_key: Key for the model prediction text.
        use_regex_prefilter: If True, try regex first and skip API for obvious answers.

    Returns:
        (results, stats) where results[i] is a letter (e.g. "B") or None,
        and stats is a dict with API call counts.
    """
    import anthropic
    async_client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    results: list[str | None] = [None] * len(samples)

    stats = {
        "total": len(samples),
        "skipped_empty": 0,
        "regex_resolved": 0,
        "api_calls": 0,
        "api_errors": 0,
        "none_answers": 0,
    }

    async def process_one(idx: int, sample: dict) -> None:
        question = sample.get(question_key, "")
        choices = sample.get(choices_key, [])
        prediction = sample.get(prediction_key, "")

        # Strip thinking blocks
        prediction = remove_thinking_process(prediction)

        if not prediction.strip():
            stats["skipped_empty"] += 1
            return

        # Regex pre-filter for obvious answers
        if use_regex_prefilter:
            regex_result = _quick_regex_extract(prediction, choices)
            if regex_result is not None:
                results[idx] = regex_result
                stats["regex_resolved"] += 1
                return

        # Claude API call
        prompt = _build_prompt(question, choices, prediction)
        async with semaphore:
            try:
                response = await async_client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=10,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                stats["api_calls"] += 1
                answer = _parse_claude_answer(response.content[0].text, choices)
                results[idx] = answer
                if answer is None:
                    stats["none_answers"] += 1
            except Exception as e:
                stats["api_errors"] += 1
                if stats["api_errors"] <= 5:
                    print(f"  Warning: API error on sample {idx}: {e}")
                elif stats["api_errors"] == 6:
                    print("  (suppressing further API error messages)")

    # Process all samples concurrently
    t0 = time.time()
    tasks = [process_one(i, s) for i, s in enumerate(samples)]
    await asyncio.gather(*tasks)
    elapsed = time.time() - t0

    stats["elapsed_seconds"] = round(elapsed, 1)
    resolved = stats["regex_resolved"] + stats["api_calls"] - stats["api_errors"]
    print(f"  Extraction complete: {resolved}/{stats['total']} resolved "
          f"({stats['regex_resolved']} regex, {stats['api_calls']} API calls, "
          f"{stats['api_errors']} errors) in {elapsed:.1f}s")

    return results, stats


def letter_to_index(letter: str) -> int:
    """Convert letter (A/B/C/D) to 0-based index."""
    return ord(letter.upper()) - ord('A')
