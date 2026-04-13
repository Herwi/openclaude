import { afterEach, describe, expect, test } from 'bun:test'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  getGeminiProjectIdHint,
  mayHaveGeminiAdcCredentials,
  resolveGeminiCredential,
} from './geminiAuth.ts'

const existingFilePath = import.meta.path

const originalEnv = {
  GEMINI_API_KEY: process.env.GEMINI_API_KEY,
  GOOGLE_API_KEY: process.env.GOOGLE_API_KEY,
  GEMINI_ACCESS_TOKEN: process.env.GEMINI_ACCESS_TOKEN,
  GEMINI_AUTH_MODE: process.env.GEMINI_AUTH_MODE,
  GOOGLE_APPLICATION_CREDENTIALS: process.env.GOOGLE_APPLICATION_CREDENTIALS,
  GOOGLE_CLOUD_PROJECT: process.env.GOOGLE_CLOUD_PROJECT,
  GCLOUD_PROJECT: process.env.GCLOUD_PROJECT,
  GOOGLE_PROJECT_ID: process.env.GOOGLE_PROJECT_ID,
  GEMINI_CLI_OAUTH_PATH: process.env.GEMINI_CLI_OAUTH_PATH,
  APPDATA: process.env.APPDATA,
}

function restoreEnv(key: string, value: string | undefined): void {
  if (value === undefined) {
    delete process.env[key]
  } else {
    process.env[key] = value
  }
}

afterEach(() => {
  restoreEnv('GEMINI_API_KEY', originalEnv.GEMINI_API_KEY)
  restoreEnv('GOOGLE_API_KEY', originalEnv.GOOGLE_API_KEY)
  restoreEnv('GEMINI_ACCESS_TOKEN', originalEnv.GEMINI_ACCESS_TOKEN)
  restoreEnv('GEMINI_AUTH_MODE', originalEnv.GEMINI_AUTH_MODE)
  restoreEnv(
    'GOOGLE_APPLICATION_CREDENTIALS',
    originalEnv.GOOGLE_APPLICATION_CREDENTIALS,
  )
  restoreEnv('GOOGLE_CLOUD_PROJECT', originalEnv.GOOGLE_CLOUD_PROJECT)
  restoreEnv('GCLOUD_PROJECT', originalEnv.GCLOUD_PROJECT)
  restoreEnv('GOOGLE_PROJECT_ID', originalEnv.GOOGLE_PROJECT_ID)
  restoreEnv('GEMINI_CLI_OAUTH_PATH', originalEnv.GEMINI_CLI_OAUTH_PATH)
  restoreEnv('APPDATA', originalEnv.APPDATA)
})

describe('resolveGeminiCredential', () => {
  test('prefers GEMINI_API_KEY over other Gemini auth inputs', async () => {
    process.env.GEMINI_API_KEY = 'gem-key'
    process.env.GOOGLE_API_KEY = 'google-key'
    process.env.GEMINI_ACCESS_TOKEN = 'token-123'

    await expect(resolveGeminiCredential(process.env)).resolves.toEqual({
      kind: 'api-key',
      credential: 'gem-key',
    })
  })

  test('uses GEMINI_ACCESS_TOKEN when no API key is configured', async () => {
    delete process.env.GEMINI_API_KEY
    delete process.env.GOOGLE_API_KEY
    process.env.GEMINI_AUTH_MODE = 'access-token'
    process.env.GEMINI_ACCESS_TOKEN = 'token-123'
    process.env.GOOGLE_CLOUD_PROJECT = 'test-project'

    await expect(resolveGeminiCredential(process.env)).resolves.toEqual({
      kind: 'access-token',
      credential: 'token-123',
      projectId: 'test-project',
    })
  })

  test('falls back to ADC when available', async () => {
    delete process.env.GEMINI_API_KEY
    delete process.env.GOOGLE_API_KEY
    delete process.env.GEMINI_ACCESS_TOKEN
    process.env.GEMINI_AUTH_MODE = 'adc'
    process.env.GOOGLE_APPLICATION_CREDENTIALS = existingFilePath

    const fakeAuth = {
      async getClient() {
        return {
          async getAccessToken() {
            return { token: 'adc-token' }
          },
        }
      },
      async getProjectId() {
        return 'adc-project'
      },
    }

    await expect(
      resolveGeminiCredential(process.env, {
        createGoogleAuth: async () => fakeAuth,
      }),
    ).resolves.toEqual({
      kind: 'adc',
      credential: 'adc-token',
      projectId: 'adc-project',
    })
  })

  test('cli-oauth mode reads Gemini CLI credentials and returns them as a cli-oauth credential', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'gemini-auth-cli-oauth-'))
    try {
      const path = join(dir, 'oauth_creds.json')
      writeFileSync(
        path,
        JSON.stringify({
          access_token: 'cli-token',
          refresh_token: 'r',
          // Far future so no refresh HTTP call is attempted.
          expiry_date: Date.now() + 24 * 60 * 60 * 1000,
        }),
      )
      delete process.env.GEMINI_API_KEY
      delete process.env.GOOGLE_API_KEY
      delete process.env.GEMINI_ACCESS_TOKEN
      // Project hint avoids a network call to loadCodeAssist.
      process.env.GOOGLE_CLOUD_PROJECT = 'proj-cli'
      process.env.GEMINI_AUTH_MODE = 'cli-oauth'
      process.env.GEMINI_CLI_OAUTH_PATH = path

      await expect(resolveGeminiCredential(process.env)).resolves.toEqual({
        kind: 'cli-oauth',
        credential: 'cli-token',
        projectId: 'proj-cli',
      })
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })

  test('cli-oauth mode returns none when credentials file is missing', async () => {
    delete process.env.GEMINI_API_KEY
    delete process.env.GOOGLE_API_KEY
    delete process.env.GEMINI_ACCESS_TOKEN
    delete process.env.GOOGLE_CLOUD_PROJECT
    process.env.GEMINI_AUTH_MODE = 'cli-oauth'
    process.env.GEMINI_CLI_OAUTH_PATH = '/definitely/not/here.json'

    await expect(resolveGeminiCredential(process.env)).resolves.toEqual({
      kind: 'none',
    })
  })

  test('returns none when no Gemini auth source is configured', async () => {
    delete process.env.GEMINI_API_KEY
    delete process.env.GOOGLE_API_KEY
    delete process.env.GEMINI_ACCESS_TOKEN
    delete process.env.GOOGLE_APPLICATION_CREDENTIALS

    await expect(resolveGeminiCredential(process.env)).resolves.toEqual({
      kind: 'none',
    })
  })

  test('access-token mode does not silently fall back to ADC', async () => {
    delete process.env.GEMINI_API_KEY
    delete process.env.GOOGLE_API_KEY
    delete process.env.GEMINI_ACCESS_TOKEN
    process.env.GEMINI_AUTH_MODE = 'access-token'
    process.env.GOOGLE_APPLICATION_CREDENTIALS = existingFilePath

    const fakeAuth = {
      async getClient() {
        return {
          async getAccessToken() {
            return { token: 'adc-token' }
          },
        }
      },
    }

    await expect(
      resolveGeminiCredential(process.env, {
        createGoogleAuth: async () => fakeAuth,
      }),
    ).resolves.toEqual({
      kind: 'none',
    })
  })

  test('adc mode ignores GEMINI_ACCESS_TOKEN and uses ADC credentials', async () => {
    delete process.env.GEMINI_API_KEY
    delete process.env.GOOGLE_API_KEY
    process.env.GEMINI_AUTH_MODE = 'adc'
    process.env.GEMINI_ACCESS_TOKEN = 'token-123'
    process.env.GOOGLE_APPLICATION_CREDENTIALS = existingFilePath

    const fakeAuth = {
      async getClient() {
        return {
          async getAccessToken() {
            return { token: 'adc-token' }
          },
        }
      },
      async getProjectId() {
        return 'adc-project'
      },
    }

    await expect(
      resolveGeminiCredential(process.env, {
        createGoogleAuth: async () => fakeAuth,
      }),
    ).resolves.toEqual({
      kind: 'adc',
      credential: 'adc-token',
      projectId: 'adc-project',
    })
  })
})

describe('Gemini auth helpers', () => {
  test('detects explicit project id hints', () => {
    process.env.GOOGLE_PROJECT_ID = 'project-a'
    expect(getGeminiProjectIdHint(process.env)).toBe('project-a')
  })

  test('only treats existing ADC paths as valid hints', () => {
    process.env.GOOGLE_APPLICATION_CREDENTIALS = existingFilePath
    expect(mayHaveGeminiAdcCredentials(process.env)).toBe(true)

    process.env.GOOGLE_APPLICATION_CREDENTIALS = `${existingFilePath}.missing`
    process.env.APPDATA = undefined
    expect(mayHaveGeminiAdcCredentials(process.env)).toBe(false)
  })
})
