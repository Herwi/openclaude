import { describe, expect, test } from 'bun:test'

import {
  geminiCodeAssistFetch,
  sanitizeSchemaForCodeAssist,
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

  test('clamps max_tokens=65536 down to 65535 (Code Assist exclusive upper bound)', () => {
    // Regression: Claude Code's Sonnet-tier default max_tokens is 65536,
    // which sits exactly on Code Assist's exclusive upper bound. Without
    // clamping, every such request is rejected with a 400 INVALID_ARGUMENT
    // before the model is even consulted.
    const gemini = translateOpenAIRequestToGemini({
      messages: [{ role: 'user', content: 'x' }],
      max_tokens: 65536,
    })
    expect(gemini.generationConfig?.maxOutputTokens).toBe(65535)
  })

  test('clamps values larger than 65535 down to 65535', () => {
    const gemini = translateOpenAIRequestToGemini({
      messages: [{ role: 'user', content: 'x' }],
      max_tokens: 200_000,
    })
    expect(gemini.generationConfig?.maxOutputTokens).toBe(65535)
  })

  test('raises zero / negative max_tokens to 1 (Code Assist inclusive lower bound)', () => {
    const gemini = translateOpenAIRequestToGemini({
      messages: [{ role: 'user', content: 'x' }],
      max_tokens: 0,
    })
    expect(gemini.generationConfig?.maxOutputTokens).toBe(1)

    const gemini2 = translateOpenAIRequestToGemini({
      messages: [{ role: 'user', content: 'x' }],
      max_tokens: -42,
    })
    expect(gemini2.generationConfig?.maxOutputTokens).toBe(1)
  })

  test('prefers max_completion_tokens over max_tokens but still clamps', () => {
    const gemini = translateOpenAIRequestToGemini({
      messages: [{ role: 'user', content: 'x' }],
      max_tokens: 100,
      max_completion_tokens: 65536,
    })
    expect(gemini.generationConfig?.maxOutputTokens).toBe(65535)
  })

  test('leaves values inside the accepted range untouched', () => {
    const gemini = translateOpenAIRequestToGemini({
      messages: [{ role: 'user', content: 'x' }],
      max_tokens: 8192,
    })
    expect(gemini.generationConfig?.maxOutputTokens).toBe(8192)
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

describe('sanitizeSchemaForCodeAssist', () => {
  // Regression for a real user report: Code Assist rejected tool schemas
  // containing "exclusiveMinimum" and "const" with 400 INVALID_ARGUMENT.

  test('strips exclusiveMinimum / exclusiveMaximum', () => {
    const out = sanitizeSchemaForCodeAssist({
      type: 'integer',
      minimum: 0,
      exclusiveMinimum: 0,
      maximum: 100,
      exclusiveMaximum: 100,
    })
    expect(out).toEqual({ type: 'integer', minimum: 0, maximum: 100 })
    expect('exclusiveMinimum' in out).toBe(false)
    expect('exclusiveMaximum' in out).toBe(false)
  })

  test('rewrites const to a single-value enum', () => {
    expect(sanitizeSchemaForCodeAssist({ const: 'foo' })).toEqual({
      enum: ['foo'],
    })
    expect(
      sanitizeSchemaForCodeAssist({ type: 'string', const: 'foo' }),
    ).toEqual({ type: 'string', enum: ['foo'] })
  })

  test('existing enum wins over const', () => {
    expect(
      sanitizeSchemaForCodeAssist({
        type: 'string',
        enum: ['a', 'b'],
        const: 'z',
      }),
    ).toEqual({ type: 'string', enum: ['a', 'b'] })
  })

  test('rewrites oneOf to anyOf', () => {
    const out = sanitizeSchemaForCodeAssist({
      oneOf: [{ const: 'foo' }, { const: 'bar' }],
    })
    expect(out).toEqual({
      anyOf: [{ enum: ['foo'] }, { enum: ['bar'] }],
    })
  })

  test('drops allOf silently', () => {
    const out = sanitizeSchemaForCodeAssist({
      type: 'object',
      allOf: [{ type: 'object', properties: { x: { type: 'number' } } }],
    })
    expect('allOf' in out).toBe(false)
    expect(out.type).toBe('object')
  })

  test('strips additionalProperties, patternProperties, $schema, $ref, default', () => {
    const out = sanitizeSchemaForCodeAssist({
      type: 'object',
      $schema: 'http://json-schema.org/draft-07/schema#',
      $ref: '#/definitions/Foo',
      default: {},
      additionalProperties: false,
      patternProperties: { '^x': { type: 'string' } },
      properties: { a: { type: 'string' } },
    })
    expect(out).toEqual({
      type: 'object',
      properties: { a: { type: 'string' } },
    })
  })

  test('strips format keyword', () => {
    expect(
      sanitizeSchemaForCodeAssist({
        type: 'string',
        format: 'path',
      }),
    ).toEqual({ type: 'string' })
  })

  test('recurses into properties', () => {
    expect(
      sanitizeSchemaForCodeAssist({
        type: 'object',
        properties: {
          count: {
            type: 'integer',
            minimum: 0,
            exclusiveMinimum: 0,
          },
          mode: { const: 'fast' },
        },
      }),
    ).toEqual({
      type: 'object',
      properties: {
        count: { type: 'integer', minimum: 0 },
        mode: { enum: ['fast'] },
      },
    })
  })

  test('recurses into array items', () => {
    expect(
      sanitizeSchemaForCodeAssist({
        type: 'array',
        items: { type: 'object', properties: { tag: { const: 'x' } } },
      }),
    ).toEqual({
      type: 'array',
      items: {
        type: 'object',
        properties: { tag: { enum: ['x'] } },
      },
    })
  })

  test('recurses into anyOf branches', () => {
    expect(
      sanitizeSchemaForCodeAssist({
        anyOf: [
          { type: 'string', const: 'fast' },
          { type: 'integer', exclusiveMinimum: 0 },
        ],
      }),
    ).toEqual({
      anyOf: [
        { type: 'string', enum: ['fast'] },
        { type: 'integer' },
      ],
    })
  })

  test('converts type arrays and null types to nullable + single type', () => {
    expect(
      sanitizeSchemaForCodeAssist({ type: ['string', 'null'] }),
    ).toEqual({ type: 'string', nullable: true })

    expect(sanitizeSchemaForCodeAssist({ type: 'null' })).toEqual({
      nullable: true,
    })
  })

  test('preserves an explicit nullable: true', () => {
    expect(
      sanitizeSchemaForCodeAssist({ type: 'string', nullable: true }),
    ).toEqual({ type: 'string', nullable: true })
  })

  test('filters required to properties that survived sanitization', () => {
    expect(
      sanitizeSchemaForCodeAssist({
        type: 'object',
        required: ['a', 'b', 'ghost'],
        properties: {
          a: { type: 'string' },
          b: { type: 'string' },
        },
      }),
    ).toEqual({
      type: 'object',
      required: ['a', 'b'],
      properties: {
        a: { type: 'string' },
        b: { type: 'string' },
      },
    })
  })

  test('passes through supported numeric/string/array constraints', () => {
    expect(
      sanitizeSchemaForCodeAssist({
        type: 'array',
        minItems: 1,
        maxItems: 10,
        items: {
          type: 'string',
          minLength: 1,
          maxLength: 50,
          pattern: '^[a-z]+$',
        },
      }),
    ).toEqual({
      type: 'array',
      minItems: 1,
      maxItems: 10,
      items: {
        type: 'string',
        minLength: 1,
        maxLength: 50,
        pattern: '^[a-z]+$',
      },
    })
  })

  test('end-to-end: sanitization flows through translateOpenAIRequestToGemini tools mapping', () => {
    // Synthesises the exact offending shape from the real user report:
    // properties with both const-inside-oneOf and exclusiveMinimum on a
    // numeric field. Previously this hit the wire unchanged and Code Assist
    // returned 400 INVALID_ARGUMENT.
    const gemini = translateOpenAIRequestToGemini({
      messages: [{ role: 'user', content: 'x' }],
      tools: [
        {
          type: 'function',
          function: {
            name: 'TestTool',
            description: 'Demonstrates hostile input',
            parameters: {
              type: 'object',
              properties: {
                count: {
                  type: 'integer',
                  minimum: 1,
                  exclusiveMinimum: 1,
                },
                mode: {
                  oneOf: [{ const: 'fast' }, { const: 'slow' }],
                },
              },
              required: ['count'],
              additionalProperties: false,
            },
          },
        },
      ],
    })
    expect(gemini.tools).toHaveLength(1)
    const decl = gemini.tools![0].functionDeclarations[0]
    expect(decl.parameters).toEqual({
      type: 'object',
      properties: {
        count: { type: 'integer', minimum: 1 },
        mode: { anyOf: [{ enum: ['fast'] }, { enum: ['slow'] }] },
      },
      required: ['count'],
    })
    // No unsupported keys anywhere in the declared schema.
    const serialized = JSON.stringify(decl.parameters)
    expect(serialized).not.toContain('exclusiveMinimum')
    expect(serialized).not.toContain('"const"')
    expect(serialized).not.toContain('oneOf')
    expect(serialized).not.toContain('additionalProperties')
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

  test('streaming parses CRLF-delimited SSE frames (real Code Assist framing)', async () => {
    // Regression test for a critical bug caught by running against a live
    // Code Assist endpoint: Google's streamGenerateContent uses CRLF line
    // endings (`\r\n\r\n` between events), not LF (`\n\n`). A previous
    // version of the stream decoder split on `\n\n` and never matched a
    // single frame, causing every streaming response to be truncated to
    // whatever the flush path happened to salvage from the first `data:`
    // line — typically one or two words.
    const encoder = new TextEncoder()
    const geminiSseBody = new ReadableStream<Uint8Array>({
      start(controller) {
        // Frame shape mirrors what Code Assist actually sends, including the
        // top-level `response` wrapper and CRLF separators.
        const frame1 = `data: ${JSON.stringify({
          response: {
            candidates: [
              {
                content: { role: 'model', parts: [{ text: 'The' }] },
              },
            ],
          },
        })}\r\n\r\n`
        const frame2 = `data: ${JSON.stringify({
          response: {
            candidates: [
              {
                content: {
                  role: 'model',
                  parts: [
                    {
                      text: ' quick brown fox jumps over the lazy dog.',
                    },
                  ],
                },
                finishReason: 'STOP',
              },
            ],
            usageMetadata: {
              promptTokenCount: 10,
              candidatesTokenCount: 10,
              totalTokenCount: 20,
            },
          },
        })}\r\n\r\n`
        // Split the stream across multiple chunks to also exercise partial-
        // frame buffering.
        const combined = frame1 + frame2
        const mid = Math.floor(combined.length / 2)
        controller.enqueue(encoder.encode(combined.slice(0, mid)))
        controller.enqueue(encoder.encode(combined.slice(mid)))
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
        model: 'gemini-2.5-flash-lite',
        body: {
          model: 'gemini-2.5-flash-lite',
          messages: [{ role: 'user', content: 'x' }],
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
    const frames = text.split('\n\n').filter(Boolean)
    const datas = frames
      .map(f => f.replace(/^data:\s*/, ''))
      .filter(d => d && d !== '[DONE]')
      .map(d => JSON.parse(d))
    const content = datas
      .flatMap(d => d.choices?.[0]?.delta?.content ?? [])
      .join('')
    // The whole sentence must survive, not just the first word.
    expect(content).toBe('The quick brown fox jumps over the lazy dog.')
    const finish = datas.find(d => d.choices?.[0]?.finish_reason)
    expect(finish?.choices?.[0]?.finish_reason).toBe('stop')
    expect(text.trim().endsWith('data: [DONE]')).toBe(true)
  })

  test('preserves Code Assist Retry-After hint on 429 so withRetry honors it', async () => {
    // Regression: free-tier Code Assist for gemini-2.5-flash-lite returns
    // 429 with a human-readable "Your quota will reset after Ns" in the
    // error message body. Previously errorResponse built a fresh Response
    // with only Content-Type, discarding any upstream Retry-After header.
    // That caused withRetry to fall back to its default exponential
    // backoff, which blows through the remaining quota before the reset
    // window elapses and compounds 429s on subsequent retries.
    const upstreamBody = JSON.stringify({
      error: {
        code: 429,
        message:
          'You have exhausted your capacity on this model. Your quota will reset after 45s.',
        status: 'RESOURCE_EXHAUSTED',
      },
    })
    const fakeFetch = (async () =>
      new Response(upstreamBody, { status: 429 })) as typeof fetch
    const response = await geminiCodeAssistFetch(
      {
        model: 'gemini-2.5-flash-lite',
        body: { messages: [{ role: 'user', content: 'x' }] },
      },
      {
        fetchImpl: fakeFetch,
        loadToken: async () => ({ accessToken: 'abc' }),
        resolveProjectId: async () => 'p',
      },
    )
    expect(response.status).toBe(429)
    expect(response.headers.get('Retry-After')).toBe('45')
  })

  test('parses minute and hour suffixes in Retry-After hints', async () => {
    const upstreamBody = JSON.stringify({
      error: { message: 'Your quota will reset after 2m.' },
    })
    const response = await geminiCodeAssistFetch(
      {
        model: 'gemini-2.5-flash-lite',
        body: { messages: [{ role: 'user', content: 'x' }] },
      },
      {
        fetchImpl: (async () =>
          new Response(upstreamBody, { status: 429 })) as typeof fetch,
        loadToken: async () => ({ accessToken: 'abc' }),
        resolveProjectId: async () => 'p',
      },
    )
    expect(response.headers.get('Retry-After')).toBe('120')
  })

  test('prefers upstream Retry-After header over parsed message', async () => {
    const response = await geminiCodeAssistFetch(
      {
        model: 'gemini-2.5-flash-lite',
        body: { messages: [{ role: 'user', content: 'x' }] },
      },
      {
        fetchImpl: (async () =>
          new Response('Your quota will reset after 999s.', {
            status: 429,
            headers: { 'Retry-After': '30' },
          })) as typeof fetch,
        loadToken: async () => ({ accessToken: 'abc' }),
        resolveProjectId: async () => 'p',
      },
    )
    expect(response.headers.get('Retry-After')).toBe('30')
  })

  test('omits Retry-After when neither header nor message hint is present', async () => {
    const response = await geminiCodeAssistFetch(
      {
        model: 'gemini-2.5-flash-lite',
        body: { messages: [{ role: 'user', content: 'x' }] },
      },
      {
        fetchImpl: (async () =>
          new Response('boom', { status: 503 })) as typeof fetch,
        loadToken: async () => ({ accessToken: 'abc' }),
        resolveProjectId: async () => 'p',
      },
    )
    expect(response.headers.get('Retry-After')).toBeNull()
  })

  test('classifies a loadCodeAssist 400 as a bad-request bug, not onboarding', async () => {
    // Regression for a real user report: the user hit a 400 INVALID_ARGUMENT
    // from loadCodeAssist because of a bad platform enum, but the transport
    // was surfacing it as `gemini_code_assist_onboarding_required` with a
    // "run gemini once and accept the terms" hint. That sent users on a
    // wild goose chase — 400s are never fixed by re-authenticating.
    const response = await geminiCodeAssistFetch(
      {
        model: 'gemini-2.5-pro',
        body: { messages: [{ role: 'user', content: 'x' }] },
      },
      {
        fetchImpl: (async () =>
          new Response('', { status: 200 })) as typeof fetch,
        loadToken: async () => ({ accessToken: 'abc' }),
        resolveProjectId: async () => {
          throw new Error(
            "Code Assist loadCodeAssist failed (400): Invalid value at 'metadata.platform' ...",
          )
        },
      },
    )
    expect(response.status).toBe(400)
    const body = (await response.json()) as {
      error?: { message?: string; type?: string }
    }
    expect(body.error?.type).toBe('gemini_code_assist_bad_request')
    expect(body.error?.message).not.toContain('accept the Code Assist terms')
    expect(body.error?.message).toContain('metadata.platform')
  })

  test('classifies loadCodeAssist 404 and missing project as onboarding', async () => {
    const response = await geminiCodeAssistFetch(
      {
        model: 'gemini-2.5-pro',
        body: { messages: [{ role: 'user', content: 'x' }] },
      },
      {
        fetchImpl: (async () =>
          new Response('', { status: 200 })) as typeof fetch,
        loadToken: async () => ({ accessToken: 'abc' }),
        resolveProjectId: async () => {
          throw new Error(
            'Code Assist did not return a cloudaicompanionProject. Run `gemini` once and complete onboarding, or set GOOGLE_CLOUD_PROJECT explicitly.',
          )
        },
      },
    )
    expect(response.status).toBe(403)
    const body = (await response.json()) as {
      error?: { type?: string }
    }
    expect(body.error?.type).toBe('gemini_code_assist_onboarding_required')
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
