# jevex

One MCP tool that returns the files a coding agent should read.

Ask *where is this actually enforced?* [TypeSafe Jev](https://www.typesafe.wiki/) scores a shortlist. The agent reads those lines instead of grepping the tree. Not a second agent. Not a `jev_evaluate` wrapper.

![Illustration: grep-and-read vs codebase_investigate](docs/assets/demo.gif)

*Illustration, not a screen recording.*

**agy + Gemini 3.8 Flash · n=16 SWE-bench Verified · 1200s cap:** 160s → **69s**, $8.74 → **$3.13**. Patch file 16/16 both arms.

Fixture packet (toy `session.py`, not the SWE run) — [`examples/session-expiry.json`](examples/session-expiry.json):

```json
{
  "source_of_truth": [{
    "path": "src/auth/session.py",
    "symbol": "assert_session_fresh",
    "range": [32, 35]
  }],
  "status": "sufficient"
}
```

You still plan and patch. jevex only answers **what to read**.

## Numbers

Same cheap agent both arms.

| | without jevex | with jevex |
| --- | ---: | ---: |
| Mean time | 160s | **69s** |
| Agent bill | $8.74 | **$3.13** |
| Hit the file to patch | 16 / 16 | 16 / 16 |
| Jev | — | ~$0.0015 / question |

Under a 90s cap the without-arm mostly returned empty JSON (timeout): 1/16 vs 11/16 finished. Separate Claude Code fixture, n=5: 6.8 files → 2.2.

Related: OpenJev, Aider's repo map, Serena, the agent's own grep. jevex is index-then-ask for any MCP agent.

## How it works

1. **Index** whatever a coding agent can already read (py, ts, go, rs, md, …).
2. **Ask** `codebase_investigate`.
3. **Jev** ranks a shortlist. The agent `Read`s those ranges.

```
you → coding agent → codebase_investigate → Jev on a shortlist → files + lines → agent reads
```

## Integration

Needs a TypeSafe `JEV_API_KEY`.

```bash
git clone https://github.com/jimmyhealer/jevex
cd jevex
pip install -e ".[mcp]"
```

```bash
claude mcp add jevex \
  -e JEV_API_KEY=ts_... \
  -e JEV_EXPLORER_REPO=/absolute/path/to/your/repo \
  -- jevex-mcp
```

Ask: *where is session expiry enforced?* It should call `codebase_investigate` first.

## CLI

```bash
cd /path/to/your/repo
jevex investigate "Where is session expiry actually enforced?"
```

`--json` prints the full packet. `--no-jev` is a lexical ablation (no key, not the product).

## Limits

Read-only. Not a patcher. Jev never sees the whole repo — only the shortlist.

MIT. [CONTRIBUTING.md](CONTRIBUTING.md) · [SECURITY.md](SECURITY.md)
