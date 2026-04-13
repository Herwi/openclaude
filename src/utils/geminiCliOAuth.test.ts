import { afterEach, describe, expect, test } from 'bun:test'
import { mkdtempSync, rmSync, writeFileSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  getGeminiCliOAuthPath,
  loadGeminiCliOAuthToken,
  readGeminiCliOAuthCredentialsFromDisk,
} from './geminiCliOAuth.ts'

const originalEnv = {
  GEMINI_CLI_OAUTH_PATH: process.env.GEMINI_CLI_OAUTH_PATH,
  GEMINI_CLI_OAUTH_CLIENT_ID: process.env.GEMINI_CLI_OAUTH_CLIENT_ID,
  GEMINI_CLI_OAUTH_CLIENT_SECRET: process.env.GEMINI_CLI_OAUTH_CLIENT_SECRET,
}

const tmpDirs: string[] = []

function makeCredsFile(
  contents: object | string,
  filename = 'oauth_creds.json',
): string {
  const dir = mkdtempSync(join(tmpdir(), 'gemini-cli-oauth-test-'))
  tmpDirs.push(dir)
  const path = join(dir, filename)
  writeFileSync(
    path,
    typeof contents === 'string' ? contents : JSON.stringify(contents),
  )
  return path
}

function restoreEnv(key: string, value: string | undefined): void {
  if (value === undefined) delete process.env[key]
  else process.env[key] = value
}

afterEach(() => {
  for (const d of tmpDirs.splice(0)) {
    try {
      rmSync(d, { recursive: true, force: true })
    } catch {
      /* ignore */
    }
  }
  restoreEnv('GEMINI_CLI_OAUTH_PATH', originalEnv.GEMINI_CLI_OAUTH_PATH)
  restoreEnv('GEMINI_CLI_OAUTH_CLIENT_ID', originalEnv.GEMINI_CLI_OAUTH_CLIENT_ID)
  restoreEnv(
    'GEMINI_CLI_OAUTH_CLIENT_SECRET',
    originalEnv.GEMINI_CLI_OAUTH_CLIENT_SECRET,
  )
})

describe('getGeminiCliOAuthPath', () => {
  test('honours GEMINI_CLI_OAUTH_PATH override', () => {
    expect(getGeminiCliOAuthPath({ GEMINI_CLI_OAUTH_PATH: '/tmp/x.json' })).toBe(
      '/tmp/x.json',
    )
  })

  test('defaults to ~/.gemini/oauth_creds.json', () => {
    const path = getGeminiCliOAuthPath({})
    expect(path.endsWith('/.gemini/oauth_creds.json')).toBe(true)
  })
})

describe('readGeminiCliOAuthCredentialsFromDisk', () => {
  test('throws a clear error when the file is missing', () => {
    expect(() =>
      readGeminiCliOAuthCredentialsFromDisk('/definitely/not/here.json'),
    ).toThrow(/not found/)
  })

  test('throws when the file is not JSON', () => {
    const path = makeCredsFile('{not json')
    expect(() => readGeminiCliOAuthCredentialsFromDisk(path)).toThrow(
      /not valid JSON/,
    )
  })

  test('throws when required fields are missing', () => {
    const path = makeCredsFile({ access_token: 'only' })
    expect(() => readGeminiCliOAuthCredentialsFromDisk(path)).toThrow(
      /missing access_token or refresh_token/,
    )
  })

  test('returns the parsed credentials when valid', () => {
    const path = makeCredsFile({
      access_token: 'a',
      refresh_token: 'r',
      expiry_date: 123,
    })
    expect(readGeminiCliOAuthCredentialsFromDisk(path)).toEqual({
      access_token: 'a',
      refresh_token: 'r',
      expiry_date: 123,
    })
  })
})

describe('loadGeminiCliOAuthToken', () => {
  test('returns the cached token when not near expiry', async () => {
    const future = Date.now() + 10 * 60 * 1000
    const path = makeCredsFile({
      access_token: 'cached',
      refresh_token: 'r',
      expiry_date: future,
    })
    const result = await loadGeminiCliOAuthToken({
      path,
      now: () => Date.now(),
      write: () => {
        throw new Error('should not refresh')
      },
    })
    expect(result.accessToken).toBe('cached')
    expect(result.expiryDate).toBe(future)
  })

  test('refreshes the token when expired and writes back to disk', async () => {
    process.env.GEMINI_CLI_OAUTH_CLIENT_ID = 'test-client-id'
    process.env.GEMINI_CLI_OAUTH_CLIENT_SECRET = 'test-client-secret'
    const past = Date.now() - 1000
    const path = makeCredsFile({
      access_token: 'old',
      refresh_token: 'rrr',
      expiry_date: past,
    })
    let writtenCreds: unknown
    const fakeFetch = (async (
      _url: string | URL | Request,
      init?: RequestInit,
    ) => {
      const body = init?.body
      expect(typeof body).toBe('string')
      expect((body as string).includes('refresh_token=rrr')).toBe(true)
      expect((body as string).includes('client_id=test-client-id')).toBe(true)
      return new Response(
        JSON.stringify({
          access_token: 'new',
          expires_in: 3600,
          token_type: 'Bearer',
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      )
    }) as typeof fetch
    const result = await loadGeminiCliOAuthToken({
      path,
      fetchImpl: fakeFetch,
      write: (creds, _path) => {
        writtenCreds = creds
      },
    })
    expect(result.accessToken).toBe('new')
    expect(writtenCreds).toMatchObject({
      access_token: 'new',
      refresh_token: 'rrr',
    })
  })

  test('refresh fails with a clear message when OAuth client env vars are missing', async () => {
    delete process.env.GEMINI_CLI_OAUTH_CLIENT_ID
    delete process.env.GEMINI_CLI_OAUTH_CLIENT_SECRET
    const path = makeCredsFile({
      access_token: 'old',
      refresh_token: 'rrr',
      expiry_date: 0,
    })
    await expect(
      loadGeminiCliOAuthToken({
        path,
        fetchImpl: (async () => new Response('', { status: 200 })) as typeof fetch,
        write: () => {},
      }),
    ).rejects.toThrow(/GEMINI_CLI_OAUTH_CLIENT_ID/)
  })

  test('throws a descriptive error when the refresh HTTP call fails', async () => {
    process.env.GEMINI_CLI_OAUTH_CLIENT_ID = 'cid'
    process.env.GEMINI_CLI_OAUTH_CLIENT_SECRET = 'csec'
    const path = makeCredsFile({
      access_token: 'old',
      refresh_token: 'rrr',
      expiry_date: 0,
    })
    const fakeFetch = (async () =>
      new Response(
        JSON.stringify({ error: 'invalid_grant' }),
        { status: 400 },
      )) as typeof fetch
    await expect(
      loadGeminiCliOAuthToken({ path, fetchImpl: fakeFetch, write: () => {} }),
    ).rejects.toThrow(/invalid_grant/)
  })

  test('persists refreshed token back to the file so gemini CLI stays in sync', async () => {
    process.env.GEMINI_CLI_OAUTH_CLIENT_ID = 'cid'
    process.env.GEMINI_CLI_OAUTH_CLIENT_SECRET = 'csec'
    const path = makeCredsFile({
      access_token: 'old',
      refresh_token: 'rrr',
      expiry_date: 0,
    })
    const fakeFetch = (async () =>
      new Response(
        JSON.stringify({ access_token: 'fresh', expires_in: 120 }),
        { status: 200 },
      )) as typeof fetch
    await loadGeminiCliOAuthToken({ path, fetchImpl: fakeFetch })
    const onDisk = JSON.parse(readFileSync(path, 'utf8'))
    expect(onDisk.access_token).toBe('fresh')
    expect(typeof onDisk.expiry_date).toBe('number')
  })
})
