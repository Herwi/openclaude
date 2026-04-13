# OpenClaude

OpenClaude is an open-source coding-agent CLI for cloud and local model providers.

Use OpenAI-compatible APIs, Gemini, GitHub Models, Codex, Ollama, Atomic Chat, and other supported backends while keeping one terminal-first workflow: prompts, tools, agents, MCP, slash commands, and streaming output.

[![PR Checks](https://github.com/Gitlawb/openclaude/actions/workflows/pr-checks.yml/badge.svg?branch=main)](https://github.com/Gitlawb/openclaude/actions/workflows/pr-checks.yml)
[![Release](https://img.shields.io/github/v/tag/Gitlawb/openclaude?label=release&color=0ea5e9)](https://github.com/Gitlawb/openclaude/tags)
[![Discussions](https://img.shields.io/badge/discussions-open-7c3aed)](https://github.com/Gitlawb/openclaude/discussions)
[![Security Policy](https://img.shields.io/badge/security-policy-0f766e)](SECURITY.md)
[![License](https://img.shields.io/badge/license-MIT-2563eb)](LICENSE)

[Quick Start](#quick-start) | [Setup Guides](#setup-guides) | [Providers](#supported-providers) | [Source Build](#source-build-and-local-development) | [VS Code Extension](#vs-code-extension) | [Community](#community)

## Why OpenClaude

- Use one CLI across cloud APIs and local model backends
- Save provider profiles inside the app with `/provider`
- Run with OpenAI-compatible services, Gemini, GitHub Models, Codex, Ollama, Atomic Chat, and other supported providers
- Keep coding-agent workflows in one place: bash, file tools, grep, glob, agents, tasks, MCP, and web tools
- Use the bundled VS Code extension for launch integration and theme support

## Quick Start

### Install

```bash
npm install -g @gitlawb/openclaude
```

If the install later reports `ripgrep not found`, install ripgrep system-wide and confirm `rg --version` works in the same terminal before starting OpenClaude.

### Start

```bash
openclaude
```

Inside OpenClaude:

- run `/provider` for guided provider setup and saved profiles
- run `/onboard-github` for GitHub Models onboarding

### Fastest OpenAI setup

macOS / Linux:

```bash
export CLAUDE_CODE_USE_OPENAI=1
export OPENAI_API_KEY=sk-your-key-here
export OPENAI_MODEL=gpt-4o

openclaude
```

Windows PowerShell:

```powershell
$env:CLAUDE_CODE_USE_OPENAI="1"
$env:OPENAI_API_KEY="sk-your-key-here"
$env:OPENAI_MODEL="gpt-4o"

openclaude
```

### Fastest local Ollama setup

macOS / Linux:

```bash
export CLAUDE_CODE_USE_OPENAI=1
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_MODEL=qwen2.5-coder:7b

openclaude
```

Windows PowerShell:

```powershell
$env:CLAUDE_CODE_USE_OPENAI="1"
$env:OPENAI_BASE_URL="http://localhost:11434/v1"
$env:OPENAI_MODEL="qwen2.5-coder:7b"

openclaude
```

## Setup Guides

Beginner-friendly guides:

- [Non-Technical Setup](docs/non-technical-setup.md)
- [Windows Quick Start](docs/quick-start-windows.md)
- [macOS / Linux Quick Start](docs/quick-start-mac-linux.md)

Advanced and source-build guides:

- [Advanced Setup](docs/advanced-setup.md)
- [Android Install](ANDROID_INSTALL.md)

## Supported Providers

| Provider | Setup Path | Notes |
| --- | --- | --- |
| OpenAI-compatible | `/provider` or env vars | Works with OpenAI, OpenRouter, DeepSeek, Groq, Mistral, LM Studio, and other compatible `/v1` servers |
| Gemini | `/provider` or env vars | Supports API key, access token, local ADC workflow, or the experimental Gemini CLI OAuth reuse mode (see [Experimental: Gemini CLI OAuth reuse](#experimental-gemini-cli-oauth-reuse)) |
| GitHub Models | `/onboard-github` | Interactive onboarding with saved credentials |
| Codex | `/provider` | Uses existing Codex credentials when available |
| Ollama | `/provider` or env vars | Local inference with no API key |
| Atomic Chat | advanced setup | Local Apple Silicon backend |
| Bedrock / Vertex / Foundry | env vars | Additional provider integrations for supported environments |

## What Works

- **Tool-driven coding workflows**: Bash, file read/write/edit, grep, glob, agents, tasks, MCP, and slash commands
- **Streaming responses**: Real-time token output and tool progress
- **Tool calling**: Multi-step tool loops with model calls, tool execution, and follow-up responses
- **Images**: URL and base64 image inputs for providers that support vision
- **Provider profiles**: Guided setup plus saved `.openclaude-profile.json` support
- **Local and remote model backends**: Cloud APIs, local servers, and Apple Silicon local inference

## Provider Notes

OpenClaude supports multiple providers, but behavior is not identical across all of them.

- Anthropic-specific features may not exist on other providers
- Tool quality depends heavily on the selected model
- Smaller local models can struggle with long multi-step tool flows
- Some providers impose lower output caps than the CLI defaults, and OpenClaude adapts where possible

For best results, use models with strong tool/function calling support.

## Agent Routing

OpenClaude can route different agents to different models through settings-based routing. This is useful for cost optimization or splitting work by model strength.

Add to `~/.claude/settings.json`:

```json
{
  "agentModels": {
    "deepseek-chat": {
      "base_url": "https://api.deepseek.com/v1",
      "api_key": "sk-your-key"
    },
    "gpt-4o": {
      "base_url": "https://api.openai.com/v1",
      "api_key": "sk-your-key"
    }
  },
  "agentRouting": {
    "Explore": "deepseek-chat",
    "Plan": "gpt-4o",
    "general-purpose": "gpt-4o",
    "frontend-dev": "deepseek-chat",
    "default": "gpt-4o"
  }
}
```

When no routing match is found, the global provider remains the fallback.

> **Note:** `api_key` values in `settings.json` are stored in plaintext. Keep this file private and do not commit it to version control.

## Web Search and Fetch

By default, `WebSearch` works on non-Anthropic models using DuckDuckGo. This gives GPT-4o, DeepSeek, Gemini, Ollama, and other OpenAI-compatible providers a free web search path out of the box.

> **Note:** DuckDuckGo fallback works by scraping search results and may be rate-limited, blocked, or subject to DuckDuckGo's Terms of Service. If you want a more reliable supported option, configure Firecrawl.

For Anthropic-native backends and Codex responses, OpenClaude keeps the native provider web search behavior.

`WebFetch` works, but its basic HTTP plus HTML-to-markdown path can still fail on JavaScript-rendered sites or sites that block plain HTTP requests.

Set a [Firecrawl](https://firecrawl.dev) API key if you want Firecrawl-powered search/fetch behavior:

```bash
export FIRECRAWL_API_KEY=your-key-here
```

With Firecrawl enabled:

- `WebSearch` can use Firecrawl's search API while DuckDuckGo remains the default free path for non-Claude models
- `WebFetch` uses Firecrawl's scrape endpoint instead of raw HTTP, handling JS-rendered pages correctly

Free tier at [firecrawl.dev](https://firecrawl.dev) includes 500 credits. The key is optional.

---

## Experimental: Gemini CLI OAuth reuse

> **Warning — local experimentation only.** This mode reuses the OAuth credentials cached on disk by Google's `gemini` CLI to drive Gemini through OpenClaude. Doing this from a third-party client is **against Google's terms of service** and must not be distributed, published, or used in production. It exists here so contributors can prove the transport layer works end-to-end against Google Code Assist with a real Google account, without a public Gemini API key. If you are not deliberately opting in to that constraint, use the regular API-key, access-token, or ADC Gemini modes instead.
>
> This path does not use `generativelanguage.googleapis.com`. It calls the Code Assist API (`cloudcode-pa.googleapis.com/v1internal`) directly, speaking native Gemini `generateContent`. Translation between OpenAI-chat wire format and Gemini happens inside `src/services/api/geminiCodeAssistTransport.ts`.

### Prerequisites

- Install and run `@google/gemini-cli` once. Complete the "Login with Google" flow so that `~/.gemini/oauth_creds.json` exists and contains a live `refresh_token`.
- Optionally set `GOOGLE_CLOUD_PROJECT` to skip the Code Assist project lookup call. If unset, the transport calls `loadCodeAssist` on first use and caches the returned `cloudaicompanionProject` for ten minutes.
- If you want OpenClaude to refresh its own access token when the cached one expires (rather than requiring you to re-run `gemini`), also export the public OAuth client values the Gemini CLI itself uses. They live in the published `@google/gemini-cli` npm package under `packages/core/src/code_assist/oauth2.ts`. Copy them into your shell:

  macOS / Linux:

  ```bash
  export GEMINI_CLI_OAUTH_CLIENT_ID=<client_id_from_gemini_cli_source>
  export GEMINI_CLI_OAUTH_CLIENT_SECRET=<client_secret_from_gemini_cli_source>
  ```

  Windows PowerShell:

  ```powershell
  $env:GEMINI_CLI_OAUTH_CLIENT_ID="<client_id_from_gemini_cli_source>"
  $env:GEMINI_CLI_OAUTH_CLIENT_SECRET="<client_secret_from_gemini_cli_source>"
  ```

  These are not stored in this repository. If both are missing and the cached token is still valid, OpenClaude will serve it without needing to refresh; if it is expired, the request fails with an actionable error instructing you to set these two env vars.

### Run

macOS / Linux:

```bash
export CLAUDE_CODE_USE_GEMINI=1
export GEMINI_AUTH_MODE=cli-oauth
# Optional — defaults to gemini-2.5-pro (the only model Code Assist free tier serves)
export GEMINI_MODEL=gemini-2.5-pro
# Optional — skips the loadCodeAssist network call
export GOOGLE_CLOUD_PROJECT=your-gcp-project-id

openclaude
```

Windows PowerShell:

```powershell
$env:CLAUDE_CODE_USE_GEMINI="1"
$env:GEMINI_AUTH_MODE="cli-oauth"
# Optional — defaults to gemini-2.5-pro (the only model Code Assist free tier serves)
$env:GEMINI_MODEL="gemini-2.5-pro"
# Optional — skips the loadCodeAssist network call
$env:GOOGLE_CLOUD_PROJECT="your-gcp-project-id"

openclaude
```

Or configure it inside the app: run `/provider`, pick **Gemini**, and choose **Gemini CLI login (experimental)** as the auth method. The wizard will detect `~/.gemini/oauth_creds.json` and persist `GEMINI_AUTH_MODE=cli-oauth` in your `.openclaude-profile.json`.

### How it works

- `src/utils/geminiCliOAuth.ts` reads `~/.gemini/oauth_creds.json`, refreshes the access token when the local `expiry_date` is past its refresh window, and writes the rotated token back to disk so the real `gemini` CLI and OpenClaude stay in sync.
- `src/services/api/geminiCodeAssistTransport.ts` intercepts the HTTP call at the OpenAI-shim fetch site, translates the OpenAI chat-completion body into Gemini-native `generateContent`, POSTs to `cloudcode-pa.googleapis.com/v1internal`, and re-wraps the response as OpenAI-shaped JSON or SSE so the rest of `openaiShim.ts` sees the same wire format it expects from any other provider.
- Tool calling works end-to-end, including multi-turn loops. The transport round-trips Gemini's `thoughtSignature` through `extra_content.google.thought_signature` so replayed `functionCall` parts pass Code Assist's anti-spoof check.
- On a 401 from Code Assist, the transport force-refreshes the OAuth token and retries once. Beyond that, it surfaces a normal OpenAI-shaped error.

### Known constraints

- Free Code Assist tier only serves `gemini-2.5-pro`. If you force a different model via `GEMINI_MODEL`, Code Assist will reject the request.
- If you have not completed Code Assist onboarding (accepting its terms inside the `gemini` CLI), `loadCodeAssist` returns no `cloudaicompanionProject`, and OpenClaude surfaces a 403 telling you to run `gemini` once and accept the terms.
- The `GEMINI_CLI_OAUTH_CLIENT_ID` / `GEMINI_CLI_OAUTH_CLIENT_SECRET` values are deliberately not shipped with OpenClaude. You must copy them from the `@google/gemini-cli` npm package yourself. This is intentional: it keeps the secret-scanner off the repo and makes the terms-of-service opt-in explicit.

Again: this is a private testing path. Do not publish forks or builds that bundle the OAuth client values or otherwise automate this behavior.

---

## Headless gRPC Server

OpenClaude can be run as a headless gRPC service, allowing you to integrate its agentic capabilities (tools, bash, file editing) into other applications, CI/CD pipelines, or custom user interfaces. The server uses bidirectional streaming to send real-time text chunks, tool calls, and request permissions for sensitive commands.

### 1. Start the gRPC Server

Start the core engine as a gRPC service on `localhost:50051`:

```bash
npm run dev:grpc
```

#### Configuration

| Variable | Default | Description |
|-----------|-------------|------------------------------------------------|
| `GRPC_PORT` | `50051` | Port the gRPC server listens on |
| `GRPC_HOST` | `localhost` | Bind address. Use `0.0.0.0` to expose on all interfaces (not recommended without authentication) |

### 2. Run the Test CLI Client

We provide a lightweight CLI client that communicates exclusively over gRPC. It acts just like the main interactive CLI, rendering colors, streaming tokens, and prompting you for tool permissions (y/n) via the gRPC `action_required` event.

In a separate terminal, run:

```bash
npm run dev:grpc:cli
```

*Note: The gRPC definitions are located in `src/proto/openclaude.proto`. You can use this file to generate clients in Python, Go, Rust, or any other language.*

---

## Source Build And Local Development

```bash
bun install
bun run build
node dist/cli.mjs
```

Helpful commands:

- `bun run dev`
- `bun test`
- `bun run test:coverage`
- `bun run security:pr-scan -- --base origin/main`
- `bun run smoke`
- `bun run doctor:runtime`
- `bun run verify:privacy`
- focused `bun test ...` runs for the areas you touch

## Testing And Coverage

OpenClaude uses Bun's built-in test runner for unit tests.

Run the full unit suite:

```bash
bun test
```

Generate unit test coverage:

```bash
bun run test:coverage
```

Open the visual coverage report:

```bash
open coverage/index.html
```

If you already have `coverage/lcov.info` and only want to rebuild the UI:

```bash
bun run test:coverage:ui
```

Use focused test runs when you only touch one area:

- `bun run test:provider`
- `bun run test:provider-recommendation`
- `bun test path/to/file.test.ts`

Recommended contributor validation before opening a PR:

- `bun run build`
- `bun run smoke`
- `bun run test:coverage` for broader unit coverage when your change affects shared runtime or provider logic
- focused `bun test ...` runs for the files and flows you changed

Coverage output is written to `coverage/lcov.info`, and OpenClaude also generates a git-activity-style heatmap at `coverage/index.html`.
## Repository Structure

- `src/` - core CLI/runtime
- `scripts/` - build, verification, and maintenance scripts
- `docs/` - setup, contributor, and project documentation
- `python/` - standalone Python helpers and their tests
- `vscode-extension/openclaude-vscode/` - VS Code extension
- `.github/` - repo automation, templates, and CI configuration
- `bin/` - CLI launcher entrypoints

## VS Code Extension

The repo includes a VS Code extension in [`vscode-extension/openclaude-vscode`](vscode-extension/openclaude-vscode) for OpenClaude launch integration, provider-aware control-center UI, and theme support.

## Security

If you believe you found a security issue, see [SECURITY.md](SECURITY.md).

## Community

- Use [GitHub Discussions](https://github.com/Gitlawb/openclaude/discussions) for Q&A, ideas, and community conversation
- Use [GitHub Issues](https://github.com/Gitlawb/openclaude/issues) for confirmed bugs and actionable feature work

## Contributing

Contributions are welcome.

For larger changes, open an issue first so the scope is clear before implementation. Helpful validation commands include:

- `bun run build`
- `bun run test:coverage`
- `bun run smoke`
- focused `bun test ...` runs for touched areas

## Disclaimer

OpenClaude is an independent community project and is not affiliated with, endorsed by, or sponsored by Anthropic.

OpenClaude originated from the Claude Code codebase and has since been substantially modified to support multiple providers and open use. "Claude" and "Claude Code" are trademarks of Anthropic PBC. See [LICENSE](LICENSE) for details.

## License

See [LICENSE](LICENSE).
