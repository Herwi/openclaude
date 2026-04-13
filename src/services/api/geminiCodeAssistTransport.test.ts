import { describe, expect, test } from 'bun:test'

import {
  geminiCodeAssistFetch,
  translateGeminiResponseToOpenAI,
  translateOpenAIRequestToGemini,
} from './geminiCodeAssistTransport.ts'

describe('translateOpenAIRequestToGemini', () => {
  test('extracts system messages into systemInstruction', () => {
    const gemini = translateOpenAIRequestToGemini({
      messages: [
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: 'Hi' },
      ],
    })
    expect(gemini.systemInstruction).toEqual({
      parts: [{ text: 'You are helpful.' }],
    })
    expect(gemini.contents).toEqual([
      { role: 'user', parts: [{ text: 'Hi' }] },
    ])
  })

  test('maps assistant tool_calls to functionCall parts', () => {
    const gemini = translateOpenAIRequestToGemini({
      messages: [
        { role: 'user', content: 'read foo' },
        {
          role: 'assistant',
          content: '',
          tool_calls: [
            {
              id: 'call_1',
              type: 'function',
              function: { name: 'Read', arguments: '{"path":"foo.txt"}' },
            },
          ],
        },
        {
          role: 'tool',
          tool_call_id: 'call_1',
          content: 'file contents here',
        },
      ],
    })
    expect(gemini.contents).toEqual([
      { role: 'user', parts: [{ text: 'read foo' }] },
      {
        role: 'model',
        parts: [
          { functionCall: { name: 'Read', args: { path: 'foo.txt' } } },
        ],
      },
      {
        role: 'user',
        parts: [
          {
            functionResponse: {
              name: 'Read',
              response: { content: 'file contents here' },
            },
          },
        ],
      },
    ])
  })

  test('forwards a real Gemini thought_signature on replayed tool_calls', () => {
    const gemini = translateOpenAIRequestToGemini({
      messages: [
        { role: 'user', content: 'read foo' },
        {
          role: 'assistant',
          content: '',
          tool_calls: [
            {
              id: 'call_1',
              type: 'function',
              function: { name: 'Read', arguments: '{}' },
              extra_content: {
                google: { thought_signature: 'real-sig-abc' },
              },
            },
          ],
        },
      ],
    })
    expect(gemini.contents[1].parts[0]).toEqual({
      functionCall: { name: 'Read', args: {} },
      thoughtSignature: 'real-sig-abc',
    })
  })

  test('drops the openaiShim placeholder thought_signature instead of forwarding it to Code Assist', () => {
    // openaiShim's convertMessages stamps "skip_thought_signature_validator"
    // when there is no real Gemini signature to replay. That magic string is
    // a bypass for Google's OpenAI-compat layer and would be rejected as
    // malformed if forwarded to native Code Assist.
    const gemini = translateOpenAIRequestToGemini({
      messages: [
        {
          role: 'assistant',
          content: '',
          tool_calls: [
            {
              id: 'call_1',
              type: 'function',
              function: { name: 'Read', arguments: '{}' },
              extra_content: {
                google: {
                  thought_signature: 'skip_thought_signature_validator',
                },
              },
            },
          ],
        },
      ],
    })
    expect(gemini.contents[0].parts[0]).toEqual({
      functionCall: { name: 'Read', args: {} },
    })
    expect((gemini.contents[0].parts[0] as Record<string, unknown>).thoughtSignature).toBeUndefined()
  })

  test('maps tools[] to Gemini functionDeclarations', () => {
    const gemini = translateOpenAIRequestToGemini({
      messages: [{ role: 'user', content: 'x' }],
      tools: [
        {
          type: 'function',
          function: {
            name: 'Read',
            description: 'Reads a file',
            parameters: {
              type: 'object',
              properties: { path: { type: 'string' } },
              required: ['path'],
            },
          },
        },
      ],
      tool_choice: 'auto',
    })
    expect(gemini.tools).toEqual([
      {
        functionDeclarations: [
          {
            name: 'Read',
            description: 'Reads a file',
            parameters: {
              type: 'object',
              properties: { path: { type: 'string' } },
              required: ['path'],
            },
          },
        ],
      },
    ])
    expect(gemini.toolConfig).toEqual({
      functionCallingConfig: { mode: 'AUTO' },
    })
  })

  test('maps generation parameters', () => {
    const gemini = translateOpenAIRequestToGemini({
      messages: [{ role: 'user', content: 'x' }],
      temperature: 0.3,
      top_p: 0.9,
      max_tokens: 1024,
    })
    expect(gemini.generationConfig).toEqual({
      temperature: 0.3,
      topP: 0.9,
      maxOutputTokens: 1024,
    })
  })

  test('coalesces consecutive same-role contents', () => {
    const gemini = translateOpenAIRequestToGemini({
      messages: [
        { role: 'user', content: 'a' },
        { role: 'user', content: 'b' },
      ],
    })
    expect(gemini.contents).toHaveLength(1)
    expect(gemini.contents[0].parts).toEqual([{ text: 'a' }, { text: 'b' }])
  })
})

describe('translateGeminiResponseToOpenAI', () => {
  test('maps a text-only candidate to an OpenAI chat completion', () => {
    const openai = translateGeminiResponseToOpenAI(
      {
        candidates: [
          {
            content: { role: 'model', parts: [{ text: 'hello ' }, { text: 'world' }] },
            finishReason: 'STOP',
          },
        ],
        usageMetadata: {
          promptTokenCount: 12,
          candidatesTokenCount: 5,
          totalTokenCount: 17,
        },
      },
      'gemini-2.5-pro',
    )
    const choice = (openai.choices as Array<Record<string, unknown>>)[0]
    const message = choice.message as Record<string, unknown>
    expect(message.content).toBe('hello world')
    expect(choice.finish_reason).toBe('stop')
    expect(
      (openai.usage as Record<string, number>).prompt_tokens,
    ).toBe(12)
  })

  test('maps functionCall candidates to OpenAI tool_calls', () => {
    const openai = translateGeminiResponseToOpenAI(
      {
        candidates: [
          {
            content: {
              role: 'model',
              parts: [
                { functionCall: { name: 'Read', args: { path: 'foo' } } },
              ],
            },
            finishReason: 'STOP',
          },
        ],
      },
      'gemini-2.5-pro',
    )
    const choice = (openai.choices as Array<Record<string, unknown>>)[0]
    const message = choice.message as Record<string, unknown>
    expect(choice.finish_reason).toBe('tool_calls')
    const toolCalls = message.tool_calls as Array<Record<string, unknown>>
    expect(toolCalls).toHaveLength(1)
    const first = toolCalls[0] as { function: { name: string; arguments: string } }
    expect(first.function.name).toBe('Read')
    expect(JSON.parse(first.function.arguments)).toEqual({ path: 'foo' })
  })

  test('surfaces Gemini thoughtSignature back through extra_content.google so the next turn can replay it', () => {
    const openai = translateGeminiResponseToOpenAI(
      {
        candidates: [
          {
            content: {
              parts: [
                {
                  functionCall: { name: 'Read', args: { path: 'foo' } },
                  thoughtSignature: 'sig-from-gemini',
                },
              ],
            },
            finishReason: 'STOP',
          },
        ],
      },
      'gemini-2.5-pro',
    )
    const choice = (openai.choices as Array<Record<string, unknown>>)[0]
    const message = choice.message as Record<string, unknown>
    const toolCalls = message.tool_calls as Array<{
      extra_content?: { google?: { thought_signature?: string } }
    }>
    expect(toolCalls[0].extra_content?.google?.thought_signature).toBe(
      'sig-from-gemini',
    )
  })

  test('round-trips a real Gemini thought_signature through both translation directions', () => {
    // Simulate a multi-turn conversation: Code Assist returns a functionCall
    // with a thoughtSignature, openaiShim forwards it back to Claude Code,
    // Claude Code includes it in the next turn's tool_calls — and the
    // outbound translator MUST place it on the corresponding Gemini part.
    const inbound = translateGeminiResponseToOpenAI(
      {
        candidates: [
          {
            content: {
              parts: [
                {
                  functionCall: { name: 'Read', args: { path: 'foo' } },
                  thoughtSignature: 'round-trip-sig',
                },
              ],
            },
            finishReason: 'STOP',
          },
        ],
      },
      'gemini-2.5-pro',
    )
    const inboundChoice = (inbound.choices as Array<Record<string, unknown>>)[0]
    const inboundToolCalls = (inboundChoice.message as { tool_calls: Array<{
      id: string
      type: 'function'
      function: { name: string; arguments: string }
      extra_content?: Record<string, unknown>
    }> }).tool_calls

    const outbound = translateOpenAIRequestToGemini({
      messages: [
        { role: 'user', content: 'read foo' },
        {
          role: 'assistant',
          content: '',
          tool_calls: inboundToolCalls,
        },
        {
          role: 'tool',
          tool_call_id: inboundToolCalls[0].id,
          content: 'file contents',
        },
      ],
    })

    const modelTurn = outbound.contents[1]
    expect(modelTurn.role).toBe('model')
    expect(modelTurn.parts[0]).toEqual({
      functionCall: { name: 'Read', args: { path: 'foo' } },
      thoughtSignature: 'round-trip-sig',
    })
  })

  test('unwraps Code Assist response wrappers', () => {
    const openai = translateGeminiResponseToOpenAI(
      {
        response: {
          candidates: [
            {
              content: { parts: [{ text: 'hi' }] },
              finishReason: 'STOP',
            },
          ],
        },
      },
      'gemini-2.5-pro',
    )
    const choice = (openai.choices as Array<Record<string, unknown>>)[0]
    expect((choice.message as Record<string, unknown>).content).toBe('hi')
  })
})

describe('geminiCodeAssistFetch', () => {
  test('non-streaming happy path translates body in and out', async () => {
    let capturedUrl: string | URL | Request | undefined
    let capturedInit: RequestInit | undefined
    const fakeFetch = (async (url: string | URL | Request, init?: RequestInit) => {
      capturedUrl = url
      capturedInit = init
      return new Response(
        JSON.stringify({
          candidates: [
            {
              content: { parts: [{ text: 'pong' }] },
              finishReason: 'STOP',
            },
          ],
          usageMetadata: { promptTokenCount: 1, candidatesTokenCount: 1 },
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      )
    }) as typeof fetch

    const response = await geminiCodeAssistFetch(
      {
        model: 'gemini-2.5-pro',
        body: {
          model: 'gemini-2.5-pro',
          messages: [{ role: 'user', content: 'ping' }],
          stream: false,
        },
      },
      {
        fetchImpl: fakeFetch,
        loadToken: async () => ({ accessToken: 'abc' }),
        resolveProjectId: async () => 'project-42',
      },
    )
    expect(response.status).toBe(200)
    const json = await response.json()
    expect(
      ((json.choices as Array<Record<string, unknown>>)[0].message as Record<
        string,
        unknown
      >).content,
    ).toBe('pong')

    expect(String(capturedUrl)).toContain(':generateContent')
    expect(String(capturedUrl)).not.toContain('streamGenerateContent')
    const sent = JSON.parse(capturedInit?.body as string)
    expect(sent.model).toBe('gemini-2.5-pro')
    expect(sent.project).toBe('project-42')
    expect(sent.request.contents[0].parts[0].text).toBe('ping')
    expect(
      (capturedInit?.headers as Record<string, string>).Authorization,
    ).toBe('Bearer abc')
  })

  test('streaming translates SSE upstream into OpenAI SSE chunks', async () => {
    const encoder = new TextEncoder()
    const geminiSseBody = new ReadableStream<Uint8Array>({
      start(controller) {
        const frame1 = `data: ${JSON.stringify({
          candidates: [{ content: { parts: [{ text: 'hello' }] } }],
        })}\n\n`
        const frame2 = `data: ${JSON.stringify({
          candidates: [
            {
              content: { parts: [{ text: ' world' }] },
              finishReason: 'STOP',
            },
          ],
          usageMetadata: {
            promptTokenCount: 2,
            candidatesTokenCount: 3,
            totalTokenCount: 5,
          },
        })}\n\n`
        controller.enqueue(encoder.encode(frame1 + frame2))
        controller.close()
      },
    })
    const fakeFetch = (async () =>
      new Response(geminiSseBody, {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
      })) as typeof fetch

    const response = await geminiCodeAssistFetch(
      {
        model: 'gemini-2.5-pro',
        body: {
          model: 'gemini-2.5-pro',
          messages: [{ role: 'user', content: 'hi' }],
          stream: true,
        },
      },
      {
        fetchImpl: fakeFetch,
        loadToken: async () => ({ accessToken: 'abc' }),
        resolveProjectId: async () => 'p',
      },
    )
    expect(response.ok).toBe(true)
    const text = await response.text()
    // Expect at minimum a role chunk, two content chunks, a finish chunk, and [DONE].
    const frames = text.split('\n\n').filter(Boolean)
    const datas = frames
      .map(f => f.replace(/^data:\s*/, ''))
      .filter(d => d && d !== '[DONE]')
      .map(d => JSON.parse(d))
    const contents = datas
      .flatMap(d => d.choices?.[0]?.delta?.content ?? [])
      .join('')
    expect(contents).toBe('hello world')
    expect(text.trim().endsWith('data: [DONE]')).toBe(true)
    const finish = datas.find(d => d.choices?.[0]?.finish_reason)
    expect(finish?.choices?.[0]?.finish_reason).toBe('stop')
  })

  test('returns an OpenAI-shaped error response when Code Assist fails', async () => {
    const fakeFetch = (async () =>
      new Response('boom', {
        status: 503,
      })) as typeof fetch
    const response = await geminiCodeAssistFetch(
      {
        model: 'gemini-2.5-pro',
        body: {
          messages: [{ role: 'user', content: 'x' }],
        },
      },
      {
        fetchImpl: fakeFetch,
        loadToken: async () => ({ accessToken: 'abc' }),
        resolveProjectId: async () => 'p',
      },
    )
    expect(response.status).toBe(503)
    const json = (await response.json()) as { error?: { message?: string } }
    expect(json.error?.message).toContain('503')
  })

  test('surfaces auth loading errors as 401', async () => {
    const response = await geminiCodeAssistFetch(
      {
        model: 'gemini-2.5-pro',
        body: { messages: [{ role: 'user', content: 'x' }] },
      },
      {
        fetchImpl: (async () => new Response('')) as typeof fetch,
        loadToken: async () => {
          throw new Error('please re-login')
        },
        resolveProjectId: async () => 'p',
      },
    )
    expect(response.status).toBe(401)
    const body = (await response.json()) as { error?: { message?: string } }
    expect(body.error?.message).toContain('please re-login')
  })

  test('rejects requests with a missing messages array without calling fetch', async () => {
    let fetchCalls = 0
    const fakeFetch = (async () => {
      fetchCalls++
      return new Response('')
    }) as typeof fetch
    const response = await geminiCodeAssistFetch(
      {
        model: 'gemini-2.5-pro',
        // @ts-expect-error: intentionally malformed to exercise the guard
        body: { stream: false },
      },
      {
        fetchImpl: fakeFetch,
        loadToken: async () => ({ accessToken: 'abc' }),
        resolveProjectId: async () => 'p',
      },
    )
    expect(response.status).toBe(400)
    expect(fetchCalls).toBe(0)
    const body = (await response.json()) as { error?: { message?: string } }
    expect(body.error?.message).toContain('messages')
  })

  test('on a 401 response, force-refreshes the OAuth token and retries once with the fresh token on the wire', async () => {
    const refreshCalls: Array<boolean | undefined> = []
    const fetchAttempts: Array<{
      authorization: string | undefined
      body: Record<string, unknown>
    }> = []
    const fakeFetch = (async (
      _url: string | URL | Request,
      init?: RequestInit,
    ) => {
      const headers = init?.headers as Record<string, string> | undefined
      fetchAttempts.push({
        authorization: headers?.Authorization ?? headers?.authorization,
        body: JSON.parse(init?.body as string),
      })
      if (fetchAttempts.length === 1) {
        return new Response(JSON.stringify({ error: 'token expired' }), {
          status: 401,
        })
      }
      return new Response(
        JSON.stringify({
          candidates: [
            { content: { parts: [{ text: 'ok' }] }, finishReason: 'STOP' },
          ],
        }),
        { status: 200 },
      )
    }) as typeof fetch

    const response = await geminiCodeAssistFetch(
      {
        model: 'gemini-2.5-pro',
        body: {
          messages: [{ role: 'user', content: 'ping' }],
        },
      },
      {
        fetchImpl: fakeFetch,
        loadToken: async opts => {
          refreshCalls.push(opts?.forceRefresh)
          return {
            accessToken: opts?.forceRefresh ? 'fresh-token' : 'stale-token',
          }
        },
        resolveProjectId: async () => 'p',
      },
    )

    expect(response.ok).toBe(true)
    expect(refreshCalls).toEqual([false, true])
    expect(fetchAttempts).toHaveLength(2)
    // First attempt used the stale cached token.
    expect(fetchAttempts[0].authorization).toBe('Bearer stale-token')
    // Retry MUST send the freshly refreshed token, not the stale one.
    expect(fetchAttempts[1].authorization).toBe('Bearer fresh-token')
    // Both attempts send the same translated payload — the body shouldn't
    // be re-translated between retries.
    expect(fetchAttempts[0].body).toEqual(fetchAttempts[1].body)
    expect((fetchAttempts[1].body as { model: string }).model).toBe(
      'gemini-2.5-pro',
    )

    const json = (await response.json()) as {
      choices: Array<{ message: { content: string } }>
    }
    expect(json.choices[0].message.content).toBe('ok')
  })

  test('surfaces a clear error when the retried request also fails', async () => {
    const fakeFetch = (async () =>
      new Response(
        JSON.stringify({ error: 'still expired' }),
        { status: 401 },
      )) as typeof fetch
    const response = await geminiCodeAssistFetch(
      {
        model: 'gemini-2.5-pro',
        body: { messages: [{ role: 'user', content: 'x' }] },
      },
      {
        fetchImpl: fakeFetch,
        loadToken: async opts => ({
          accessToken: opts?.forceRefresh ? 'fresh' : 'stale',
        }),
        resolveProjectId: async () => 'p',
      },
    )
    expect(response.status).toBe(401)
  })

  test('generates tool_call ids that do not collide across parallel calls', async () => {
    const openai = translateGeminiResponseToOpenAI(
      {
        candidates: [
          {
            content: {
              parts: [
                { functionCall: { name: 'A', args: { x: 1 } } },
                { functionCall: { name: 'B', args: { x: 2 } } },
                { functionCall: { name: 'C', args: { x: 3 } } },
              ],
            },
            finishReason: 'STOP',
          },
        ],
      },
      'gemini-2.5-pro',
    )
    const choice = (openai.choices as Array<Record<string, unknown>>)[0]
    const toolCalls = (choice.message as { tool_calls: Array<{ id: string }> })
      .tool_calls
    const ids = toolCalls.map(t => t.id)
    expect(new Set(ids).size).toBe(ids.length)
    for (const id of ids) {
      expect(id).toMatch(/^call_[0-9a-f]{16,}$/)
    }
  })
})
