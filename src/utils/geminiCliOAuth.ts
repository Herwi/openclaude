// EXPERIMENTAL / LOCAL-ONLY — NOT FOR DISTRIBUTION.
//
// This module reads and refreshes the OAuth credentials cached on disk by
// Google's `gemini` CLI (the public `@google/gemini-cli` package). Those
// credentials belong to the user's Google account and are obtained via the
// "Login with Google" flow. They authorize access to the Code Assist API
// (`cloudcode-pa.googleapis.com`) — not the public Gemini API. Using them to
// power a third-party client like openclaude is against Google's terms of
// service for redistribution and exists here only for private experimentation.
//
// To refresh an expired access token this module needs the OAuth client_id
// and client_secret that the Gemini CLI itself uses. Those values are not
// stored in this repository — they must be supplied at runtime via the env
// vars GEMINI_CLI_OAUTH_CLIENT_ID and GEMINI_CLI_OAUTH_CLIENT_SECRET. They
// are embedded verbatim in the published `@google/gemini-cli` npm package
// (`packages/core/src/code_assist/oauth2.ts`) and can be copied from there
// into a local shell profile by the user. If both env vars are missing and
// the cached token is still valid, the module will still serve that token
// without needing to refresh.

import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

import { memoizeWithTTLAsync } from './memoize.js'

const GOOGLE_OAUTH_TOKEN_URL = 'https://oauth2.googleapis.com/token'
const CODE_ASSIST_ENDPOINT = 'https://cloudcode-pa.googleapis.com'

function getOAuthClientCredentials(
  env: NodeJS.ProcessEnv = process.env,
): { clientId: string; clientSecret: string } | undefined {
  const clientId = env.GEMINI_CLI_OAUTH_CLIENT_ID?.trim()
  const clientSecret = env.GEMINI_CLI_OAUTH_CLIENT_SECRET?.trim()
  if (clientId && clientSecret) return { clientId, clientSecret }
  return undefined
}

const PROJECT_CACHE_TTL_MS = 10 * 60 * 1000
const REFRESH_WINDOW_MS = 60 * 1000

export type GeminiCliOAuthCredentials = {
  access_token: string
  refresh_token: string
  scope?: string
  token_type?: string
  id_token?: string
  expiry_date?: number
}

export type GeminiCliToken = {
  accessToken: string
  expiryDate?: number
}

export function getGeminiCliOAuthPath(
  env: NodeJS.ProcessEnv = process.env,
): string {
  const override = env.GEMINI_CLI_OAUTH_PATH?.trim()
  if (override) return override
  return join(homedir(), '.gemini', 'oauth_creds.json')
}

export function readGeminiCliOAuthCredentialsFromDisk(
  path: string = getGeminiCliOAuthPath(),
): GeminiCliOAuthCredentials {
  if (!existsSync(path)) {
    throw new Error(
      `Gemini CLI OAuth credentials not found at ${path}. Install and run the \`gemini\` CLI once ("Login with Google") before enabling GEMINI_AUTH_MODE=cli-oauth.`,
    )
  }
  let raw: string
  try {
    raw = readFileSync(path, 'utf8')
  } catch (err) {
    throw new Error(
      `Failed to read Gemini CLI OAuth credentials at ${path}: ${(err as Error).message}`,
    )
  }
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    throw new Error(
      `Gemini CLI OAuth credentials at ${path} are not valid JSON. Re-run \`gemini\` to refresh them.`,
    )
  }
  if (
    !parsed ||
    typeof parsed !== 'object' ||
    typeof (parsed as GeminiCliOAuthCredentials).access_token !== 'string' ||
    typeof (parsed as GeminiCliOAuthCredentials).refresh_token !== 'string'
  ) {
    throw new Error(
      `Gemini CLI OAuth credentials at ${path} are missing access_token or refresh_token. Re-run \`gemini\` to refresh them.`,
    )
  }
  return parsed as GeminiCliOAuthCredentials
}

function writeGeminiCliOAuthCredentialsToDisk(
  creds: GeminiCliOAuthCredentials,
  path: string = getGeminiCliOAuthPath(),
): void {
  try {
    writeFileSync(path, JSON.stringify(creds, null, 2), { mode: 0o600 })
  } catch {
    // Non-fatal: if we can't write back, the next call will just refresh again.
  }
}

type RefreshResponse = {
  access_token?: string
  expires_in?: number
  token_type?: string
  id_token?: string
  scope?: string
  error?: string
  error_description?: string
}

async function refreshGeminiCliAccessToken(
  creds: GeminiCliOAuthCredentials,
  fetchImpl: typeof fetch = fetch,
  env: NodeJS.ProcessEnv = process.env,
): Promise<GeminiCliOAuthCredentials> {
  const oauthClient = getOAuthClientCredentials(env)
  if (!oauthClient) {
    throw new Error(
      'Cannot refresh the Gemini CLI OAuth token: set GEMINI_CLI_OAUTH_CLIENT_ID and GEMINI_CLI_OAUTH_CLIENT_SECRET to the public OAuth client values used by the Gemini CLI (see packages/core/src/code_assist/oauth2.ts in the @google/gemini-cli npm package).',
    )
  }
  const body = new URLSearchParams({
    client_id: oauthClient.clientId,
    client_secret: oauthClient.clientSecret,
    refresh_token: creds.refresh_token,
    grant_type: 'refresh_token',
  })
  const response = await fetchImpl(GOOGLE_OAUTH_TOKEN_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: body.toString(),
  })
  const text = await response.text()
  let parsed: RefreshResponse = {}
  try {
    parsed = JSON.parse(text) as RefreshResponse
  } catch {
    // fall through; handled below
  }
  if (!response.ok || !parsed.access_token) {
    const detail =
      parsed.error_description ?? parsed.error ?? text.slice(0, 200) ?? 'unknown error'
    throw new Error(
      `Failed to refresh Gemini CLI OAuth token (${response.status}): ${detail}. Re-run \`gemini\` to re-authenticate.`,
    )
  }
  const now = Date.now()
  const expiresInMs =
    typeof parsed.expires_in === 'number' ? parsed.expires_in * 1000 : 60 * 60 * 1000
  return {
    ...creds,
    access_token: parsed.access_token,
    token_type: parsed.token_type ?? creds.token_type,
    id_token: parsed.id_token ?? creds.id_token,
    scope: parsed.scope ?? creds.scope,
    expiry_date: now + expiresInMs,
  }
}

type LoadGeminiCliTokenDeps = {
  path?: string
  fetchImpl?: typeof fetch
  now?: () => number
  write?: (creds: GeminiCliOAuthCredentials, path: string) => void
  /**
   * When true, ignore the cached `expiry_date` and always hit the OAuth token
   * refresh endpoint. Used by the transport's 401-retry path when the server
   * has already invalidated a token the local cache still thinks is fresh.
   */
  forceRefresh?: boolean
}

export async function loadGeminiCliOAuthToken(
  deps: LoadGeminiCliTokenDeps = {},
): Promise<GeminiCliToken> {
  const path = deps.path ?? getGeminiCliOAuthPath()
  const now = deps.now ?? Date.now
  const write = deps.write ?? writeGeminiCliOAuthCredentialsToDisk
  const creds = readGeminiCliOAuthCredentialsFromDisk(path)
  const expiry = typeof creds.expiry_date === 'number' ? creds.expiry_date : 0
  if (!deps.forceRefresh && expiry - REFRESH_WINDOW_MS > now()) {
    return { accessToken: creds.access_token, expiryDate: expiry }
  }
  const refreshed = await refreshGeminiCliAccessToken(creds, deps.fetchImpl)
  write(refreshed, path)
  return { accessToken: refreshed.access_token, expiryDate: refreshed.expiry_date }
}

function getProjectIdHintFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): string | undefined {
  return (
    env.GOOGLE_CLOUD_PROJECT?.trim() ||
    env.GCLOUD_PROJECT?.trim() ||
    env.GOOGLE_PROJECT_ID?.trim() ||
    undefined
  )
}

type LoadCodeAssistResponse = {
  cloudaicompanionProject?: string
  allowedTiers?: Array<{ id?: string; isDefault?: boolean; name?: string }>
  currentTier?: { id?: string; name?: string }
}

/**
 * Map the current Node `process.platform` + `process.arch` to a value Google's
 * Code Assist Platform enum accepts. Valid values are defined in the public
 * `@google/gemini-cli` package and on the server as:
 *
 *   PLATFORM_UNSPECIFIED | DARWIN_AMD64 | DARWIN_ARM64 |
 *   LINUX_AMD64 | LINUX_ARM64 | WINDOWS_AMD64 | WINDOWS_ARM64
 *
 * Sending anything else produces a 400 INVALID_ARGUMENT from the server.
 */
export function getCodeAssistPlatform(
  proc: { platform: NodeJS.Platform; arch: string } = process,
): string {
  const archSuffix = proc.arch === 'arm64' ? 'ARM64' : 'AMD64'
  switch (proc.platform) {
    case 'darwin':
      return `DARWIN_${archSuffix}`
    case 'win32':
      return `WINDOWS_${archSuffix}`
    case 'linux':
      return `LINUX_${archSuffix}`
    default:
      return 'PLATFORM_UNSPECIFIED'
  }
}

async function callLoadCodeAssist(
  accessToken: string,
  fetchImpl: typeof fetch,
): Promise<LoadCodeAssistResponse> {
  const response = await fetchImpl(
    `${CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${accessToken}`,
      },
      body: JSON.stringify({
        metadata: {
          ideType: 'IDE_UNSPECIFIED',
          platform: getCodeAssistPlatform(),
          pluginType: 'GEMINI',
        },
      }),
    },
  )
  const text = await response.text()
  if (!response.ok) {
    // Don't lie about the cause. 401/403/404 are typically auth / onboarding
    // and benefit from the "run gemini once" hint, but a 400 is almost
    // always a client bug (bad enum, wrong shape) and telling the user to
    // re-authenticate would send them on a wild goose chase.
    const onboardingHint =
      response.status === 401 ||
      response.status === 403 ||
      response.status === 404
        ? ' Run `gemini` once and complete Code Assist onboarding if you have not already.'
        : ''
    throw new Error(
      `Code Assist loadCodeAssist failed (${response.status}): ${text.slice(0, 300)}.${onboardingHint}`,
    )
  }
  try {
    return JSON.parse(text) as LoadCodeAssistResponse
  } catch {
    throw new Error(
      `Code Assist loadCodeAssist returned non-JSON response: ${text.slice(0, 200)}`,
    )
  }
}

async function resolveGeminiCliProjectIdUncached(
  accessToken: string,
  hint: string | undefined,
  fetchImpl: typeof fetch,
): Promise<string> {
  if (hint) return hint
  const payload = await callLoadCodeAssist(accessToken, fetchImpl)
  const project = payload.cloudaicompanionProject?.trim()
  if (project) return project
  throw new Error(
    'Code Assist did not return a cloudaicompanionProject. Run `gemini` once and complete onboarding, or set GOOGLE_CLOUD_PROJECT explicitly.',
  )
}

const memoizedResolveProjectId = memoizeWithTTLAsync(
  async (accessToken: string, hint: string | undefined) =>
    resolveGeminiCliProjectIdUncached(accessToken, hint, fetch),
  PROJECT_CACHE_TTL_MS,
)

export async function resolveGeminiCliProjectId(
  accessToken: string,
  deps: { env?: NodeJS.ProcessEnv; fetchImpl?: typeof fetch } = {},
): Promise<string> {
  const hint = getProjectIdHintFromEnv(deps.env)
  if (hint) return hint
  if (deps.fetchImpl) {
    return resolveGeminiCliProjectIdUncached(accessToken, undefined, deps.fetchImpl)
  }
  return memoizedResolveProjectId(accessToken, undefined)
}

export function clearGeminiCliProjectIdCache(): void {
  memoizedResolveProjectId.cache.clear()
}

export const __internal = {
  GOOGLE_OAUTH_TOKEN_URL,
  CODE_ASSIST_ENDPOINT,
  refreshGeminiCliAccessToken,
  getOAuthClientCredentials,
}
