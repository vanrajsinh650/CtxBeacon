# CtxBeacon

> **See what your agent kept—and what it lost.**

CtxBeacon is a local-first AI agent harness that reveals what context compaction keeps or loses, detects potentially missing instructions, and tracks provider failovers.

---

## The Problem

AI agents running multi-turn coding and refactoring workflows accumulate massive context windows. When conversation history grows too large, older turns are compacted and replaced by shorter summaries.

Critical developer instructions—such as `"Never modify files under migrations/"` or security constraints—can quietly disappear from the compacted summary. The agent continues operating with an incomplete mental model, only for the developer to discover much later that a requirement was broken or the wrong file was overwritten.

CtxBeacon audits the entire lifecycle:
- Original turn-by-turn messages
- Effective compacted context
- Summaries that replaced older history
- Retained and dropped message IDs
- Post-compaction tool actions and constraint violations

### The Four Questions CtxBeacon Answers

1. **Which instruction or message disappeared?**
2. **What summary replaced the original context?**
3. **Did a provider fallback receive the same effective conversation context?**
4. **Which model actually answered the request?**

---

## Primary Feature: Compaction Diff

When context compaction triggers, CtxBeacon compares the conversation state immediately before and after summarization:

```text
TURN 1  CONTEXT 177% (142/80 tok) -> COMPACT 122% (98/80 tok)

  RETAINED
    msg-001
    sum-6a3824bc
    msg-008
    sum-e7f8e5e8

  SUMMARIZED / DROPPED
    msg-006
    msg-007

  NEW SUMMARY
    Refactored session authentication to support OAuth2 tokens. Updated token generation and session storage mechanisms.

  POSSIBLY MISSING
    "Never modify files under migrations/"

  EVIDENCE
    Original message: msg-002
    Compaction event: evt-7372f938f01f
    Keyword match: absent
    Snapshot: not enabled (original message content was not saved to disk; IDs and hashes recorded only)
    Details: Violation observed: A tool call wrote to prohibited path 'migrations/0002_update_auth.py'. Possible cause: the original requirement was present before compaction but not found by the post-compaction keyword check. A later tool call wrote to the prohibited path.

  Note: absence detected by deterministic keyword matching;
        this does not prove that the model forgot the instruction.
```

---

## Core Architecture

- **Append-only JSONL Ledger (`ledger.jsonl`):** Every run lifecycle event (`run_started`, `request_sent`, `provider_response`, `tool_call_started`, `tool_call_completed`, `compaction_started`, `compaction_completed`, `provider_failover`, `constraint_check`, `run_completed`, `request_blocked`) is validated and stored sequentially.
- **Deterministic Cryptographic Hashes:**
  - `context_hash`: SHA-256 over deterministic serialization of conversation messages.
  - `request_hash`: SHA-256 over application-side request payload sent to the provider, strictly excluding authorization credentials.
  - *Limitation Note:* A hash verifies the request payload constructed by CtxBeacon at the application boundary; it does not independently prove receipt or attention by a remote model.
- **Restricted Sandbox Demo Tools:**
  - `list_files`, `read_file`, `write_file`
  - All operations are strictly jailed to `examples/demo_workspace/` or the configured workspace. Path traversal (`../`), symlink escapes, and arbitrary system file access are rejected with `request_blocked` events. No unrestricted shell commands are executed.
- **Context Budget & Token Approximation:**
  - Token counts are estimated using `tiktoken` with `cl100k_base` and recorded as `token_count_method: "tiktoken_cl100k_approximation"`.
  - When the provider reports usage, it is recorded separately as `provider_usage_tokens` (never mixed or overwritten).
- **Deterministic Constraint Checker:**
  - Audits explicit task requirements against post-compaction context and tool calls.
  - Evaluates statuses: `preserved`, `possibly_missing`, `possible_violation`, `violation_observed`, `unable_to_verify`.
- **Provider Failover:**
  - Handles transient HTTP 429 rate limits, respects `Retry-After` headers, uses bounded retries, and switches to fallback providers while recording context hash parity.

---

## Privacy & Snapshot Policy

Full context snapshots are **opt-in** via `--snapshot`:

> [!WARNING]
> Snapshots store conversation context and tool outputs on disk under `ledger/snapshots/`. Snapshots may contain private prompts, code, or sensitive data.
>
> - `ledger/snapshots/` is strictly excluded from Git tracking via `.gitignore`.
> - API keys and credentials are never stored in snapshots, ledgers, or hashes.
> - When snapshots are disabled, CtxBeacon records safe message IDs and cryptographic hashes only, and diffs state honestly that original message content was not saved.

---

## Deterministic Keyword-Check Limitation

> [!NOTE]
> Keyword matching cannot detect a paraphrased summary that preserves a constraint using different words. It is a deterministic diagnostic, not a semantic proof that the model remembers the requirement.

---

## Mock vs. Real Runs

- **Simulation Mode (`simulation: true`):** All local development, unit tests, and the primary demonstration scenarios run deterministically using mock providers without requiring external network access or credentials.
- **Real Mode (`simulation: false`):** Runs genuine API calls against BharatCode's hosted `qwen-3.8-27b` model via `POST /responses` only when `BHARATCODE_API_KEY` is explicitly configured.

---

## Quickstart & Installation

### 1. Set Up Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Run the Unit Test Suite (Offline, No API Key Needed)

```bash
pytest -v
```

All 45 automated tests execute locally using deterministic fixtures.

---

## CLI Commands

### 1. Run the Primary Demonstration (Scenario B: Simulated Compaction Loss)

Demonstrates CtxBeacon detecting a dropped constraint in compaction followed by a prohibited file modification:

```bash
ctxbeacon demo --scenario compaction-loss
```

### 2. Run the Clean Compaction Scenario

Demonstrates normal context compaction where constraints are preserved:

```bash
ctxbeacon demo --scenario clean
```

### 3. Validate Ledger Integrity

Scan any JSONL ledger for schema compliance and syntax errors:

```bash
ctxbeacon validate --ledger ledger/ledger_simulated_loss.jsonl
```

### 4. Inspect Compaction Diff

```bash
ctxbeacon diff --ledger ledger/ledger_simulated_loss.jsonl
```

### 5. Run Real BharatCode Compaction Experiments (Scenario A)

Once `BHARATCODE_API_KEY` is configured:

```bash
export BHARATCODE_API_KEY="your_api_key_here"
ctxbeacon demo --scenario real-experiments
```

*(If credentials are not configured, the runner reports pending status with exact configuration steps rather than fabricating fake results.)*

---

## Verification & Test Results

```text
============================== 45 passed in 0.30s ==============================
- tests/test_schema.py: Event schema validation, mandatory fields, deterministic hashing, secret isolation (7 passed)
- tests/test_ledger.py: Append-only ledger writing, line-numbered parse errors, file validation (5 passed)
- tests/test_agent.py: Basic agent loop, sandboxed tools, path traversal blocks, turn limits (5 passed)
- tests/test_tokens.py: Tiktoken cl100k approximation, budget thresholds, variance tracking (5 passed)
- tests/test_compaction.py: Compaction transformations, ID stability, snapshot policies (2 passed)
- tests/test_render.py: Diff formatting, snapshot availability, error handling (4 passed)
- tests/test_constraints.py: Presence checks, violation checks, non-violation on rejected calls (5 passed)
- tests/test_fallback.py: Simulated 429 retries, Retry-After compliance, failover logging (2 passed)
- tests/test_bharatcode.py: BharatCode adapter, responses API parsing, 401/429 handling, secret isolation (5 passed)
- tests/test_scenarios.py: Scenario A runner, Scenario B loss simulation, CLI demo subcommands (5 passed)
```

---

## Current Limitations

- **Standalone Harness:** CtxBeacon owns its own agent loop. It does not inspect background processes inside proprietary closed IDE extensions like Cursor or Claude Code.
- **Lexical Checking:** Constraint loss detection uses auditable keyword rules rather than an opaque secondary LLM.
- **Approximation Variance:** Token counts use `cl100k_base` approximation; actual token counts from `qwen-3.8-27b` may vary slightly and are tracked separately when provider usage is reported.
