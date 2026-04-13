// Transport adapter that translates OpenAI chat-completion requests/responses
// to Gemini-native generateContent over Google's Code Assist API
// (`cloudcode-pa.googleapis.com/v1internal`). This lets openclaude reuse the
// OAuth credentials cached on disk by the `gemini` CLI — i.e. a user's "Login
// with Google" Gemini subscription — without touching the public Gemini API.
//
// EXPERIMENTAL / LOCAL-ONLY. See `src/utils/geminiCliOAuth.ts` for the legal
// and terms-of-service caveats.
//
// Design: takes an already-built OpenAI chat-completion request body, rewrites
// it into a Code Assist payload, POSTs it, and returns a `Response` whose body
// is OpenAI-shaped JSON (non-streaming) or OpenAI-shaped SSE (streaming). The
// rest of `openaiShim.ts` therefore does not need to know a different wire
// format is in use — its existing parser can decode the response.

import {
  loadGeminiCliOAuthToken,
  resolveGeminiCliProjectId,
} from '../../utils/geminiCliOAuth.js'

const CODE_ASSIST_ENDPOINT = 'https://cloudcode-pa.googleapis.com'

type OpenAIContentPart =
  | { type: 'text'; text?: string }
  | { type: 'image_url'; image_url?: { url?: string } }

type OpenAIMessageLike = {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content?: string | OpenAIContentPart[] | null
  tool_calls?: Array<{
    id: string
    type?: 'function'
    function: { name: string; arguments: string }
  }>
  tool_call_id?: string
  name?: string
}

type OpenAIToolLike = {
  type: 'function'
  function: {
    name: string
    description?: string
    parameters?: Record<string, unknown>
  }
}

type OpenAIRequestBody = {
  model?: string
  messages: OpenAIMessageLike[]
  tools?: OpenAIToolLike[]
  tool_choice?: unknown
  temperature?: number
  top_p?: number
  max_tokens?: number
  max_completion_tokens?: number
  stream?: boolean
}

type GeminiPart =
  | { text: string }
  | { inlineData: { mimeType: string; data: string } }
  | { functionCall: { name: string; args: Record<string, unknown> } }
  | { functionResponse: { name: string; response: Record<string, unknown> } }

type GeminiContent = {
  role: 'user' | 'model'
  parts: GeminiPart[]
}

type GeminiTool = {
  functionDeclarations: Array<{
    name: string
    description?: string
    parameters?: Record<string, unknown>
  }>
}

type GeminiToolConfig = {
  functionCallingConfig?: {
    mode?: 'AUTO' | 'ANY' | 'NONE'
    allowedFunctionNames?: string[]
  }
}

type GeminiGenerationConfig = {
  temperature?: number
  topP?: number
  maxOutputTokens?: number
  candidateCount?: number
}

type GeminiRequest = {
  contents: GeminiContent[]
  systemInstruction?: { parts: Array<{ text: string }> }
  tools?: GeminiTool[]
  toolConfig?: GeminiToolConfig
  generationConfig?: GeminiGenerationConfig
}

type CodeAssistEnvelope = {
  model: string
  project: string
  request: GeminiRequest
}

export type GeminiCodeAssistDeps = {
  loadToken?: typeof loadGeminiCliOAuthToken
  resolveProjectId?: typeof resolveGeminiCliProjectId
  fetchImpl?: typeof fetch
  endpoint?: string
}

/**
 * Parse a data URL of the form `data:<mime>;base64,<data>` into its parts, or
 * return `undefined` if the string is not a base64 data URL.
 */
function parseDataUrl(
  url: string,
): { mimeType: string; data: string } | undefined {
  if (!url.startsWith('data:')) return undefined
  const match = /^data:([^;,]+);base64,(.*)$/.exec(url)
  if (!match) return undefined
  return { mimeType: match[1], data: match[2] }
}

function openAIPartsToGeminiParts(
  content: OpenAIMessageLike['content'],
): GeminiPart[] {
  if (content == null) return []
  if (typeof content === 'string') {
    return content ? [{ text: content }] : []
  }
  const parts: GeminiPart[] = []
  for (const block of content) {
    if (block.type === 'text' && typeof block.text === 'string' && block.text) {
      parts.push({ text: block.text })
    } else if (block.type === 'image_url' && block.image_url?.url) {
      const parsed = parseDataUrl(block.image_url.url)
      if (parsed) {
        parts.push({ inlineData: parsed })
      } else {
        parts.push({ text: `[image: ${block.image_url.url}]` })
      }
    }
  }
  return parts
}

function stringToRecord(raw: string | undefined): Record<string, unknown> {
  if (!raw) return {}
  try {
    const parsed = JSON.parse(raw)
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>
    }
    return { value: parsed }
  } catch {
    return { value: raw }
  }
}

function toolResultToRecord(
  content: OpenAIMessageLike['content'],
): Record<string, unknown> {
  const text =
    typeof content === 'string'
      ? content
      : openAIPartsToGeminiParts(content)
          .map(p => ('text' in p ? p.text : ''))
          .join('')
  try {
    const parsed = JSON.parse(text)
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>
    }
    return { content: parsed }
  } catch {
    return { content: text }
  }
}

/**
 * Translate an OpenAI chat-completion request body into a Gemini generateContent
 * request body. The `systemInstruction` is extracted from the first system
 * message and tool roles are mapped to Gemini's `functionCall` / `functionResponse`
 * parts.
 */
export function translateOpenAIRequestToGemini(
  body: OpenAIRequestBody,
): GeminiRequest {
  const contents: GeminiContent[] = []
  let systemInstruction: GeminiRequest['systemInstruction']

  // Tool-call name lookup: Gemini's functionResponse parts need a `name`, but
  // OpenAI's tool-role messages only have `tool_call_id`. Walk backwards to
  // find the matching tool_call emitted by a prior assistant turn.
  const toolCallIdToName = new Map<string, string>()

  for (const msg of body.messages) {
    if (msg.role === 'system') {
      const text =
        typeof msg.content === 'string'
          ? msg.content
          : openAIPartsToGeminiParts(msg.content)
              .map(p => ('text' in p ? p.text ?? '' : ''))
              .join('')
      if (text) {
        systemInstruction = systemInstruction
          ? { parts: [...systemInstruction.parts, { text }] }
          : { parts: [{ text }] }
      }
      continue
    }

    if (msg.role === 'user') {
      const parts = openAIPartsToGeminiParts(msg.content)
      if (parts.length > 0) {
        contents.push({ role: 'user', parts })
      }
      continue
    }

    if (msg.role === 'assistant') {
      const parts: GeminiPart[] = []
      const textParts = openAIPartsToGeminiParts(msg.content)
      parts.push(...textParts)
      if (msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          toolCallIdToName.set(tc.id, tc.function.name)
          parts.push({
            functionCall: {
              name: tc.function.name,
              args: stringToRecord(tc.function.arguments),
            },
          })
        }
      }
      if (parts.length > 0) {
        contents.push({ role: 'model', parts })
      }
      continue
    }

    if (msg.role === 'tool') {
      const name =
        msg.name ??
        (msg.tool_call_id ? toolCallIdToName.get(msg.tool_call_id) : undefined) ??
        'tool'
      contents.push({
        role: 'user',
        parts: [
          {
            functionResponse: {
              name,
              response: toolResultToRecord(msg.content),
            },
          },
        ],
      })
    }
  }

  // Coalesce consecutive same-role contents — Gemini requires user/model
  // alternation (functionResponse is sent under role: 'user').
  const coalesced: GeminiContent[] = []
  for (const c of contents) {
    const prev = coalesced[coalesced.length - 1]
    if (prev && prev.role === c.role) {
      prev.parts.push(...c.parts)
    } else {
      coalesced.push(c)
    }
  }

  const request: GeminiRequest = { contents: coalesced }
  if (systemInstruction) request.systemInstruction = systemInstruction

  if (body.tools && body.tools.length > 0) {
    request.tools = [
      {
        functionDeclarations: body.tools.map(t => {
          const decl: GeminiTool['functionDeclarations'][number] = {
            name: t.function.name,
          }
          if (t.function.description) decl.description = t.function.description
          if (t.function.parameters) decl.parameters = t.function.parameters
          return decl
        }),
      },
    ]
    const tc = body.tool_choice
    if (typeof tc === 'string') {
      if (tc === 'auto')
        request.toolConfig = { functionCallingConfig: { mode: 'AUTO' } }
      else if (tc === 'required')
        request.toolConfig = { functionCallingConfig: { mode: 'ANY' } }
      else if (tc === 'none')
        request.toolConfig = { functionCallingConfig: { mode: 'NONE' } }
    } else if (tc && typeof tc === 'object' && 'function' in tc) {
      const name = (tc as { function?: { name?: string } }).function?.name
      if (name) {
        request.toolConfig = {
          functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [name] },
        }
      }
    }
  }

  const generationConfig: GeminiGenerationConfig = {}
  if (typeof body.temperature === 'number')
    generationConfig.temperature = body.temperature
  if (typeof body.top_p === 'number') generationConfig.topP = body.top_p
  const maxTokens = body.max_completion_tokens ?? body.max_tokens
  if (typeof maxTokens === 'number') generationConfig.maxOutputTokens = maxTokens
  if (Object.keys(generationConfig).length > 0) {
    request.generationConfig = generationConfig
  }

  return request
}

type GeminiCandidate = {
  content?: {
    role?: string
    parts?: Array<{
      text?: string
      functionCall?: { name?: string; args?: Record<string, unknown> }
    }>
  }
  finishReason?: string
}

type GeminiGenerateContentResponse = {
  candidates?: GeminiCandidate[]
  usageMetadata?: {
    promptTokenCount?: number
    candidatesTokenCount?: number
    totalTokenCount?: number
    cachedContentTokenCount?: number
  }
  // Code Assist wraps responses in a `response` field:
  response?: GeminiGenerateContentResponse
}

function unwrap(
  response: GeminiGenerateContentResponse,
): GeminiGenerateContentResponse {
  return response.response ?? response
}

function mapFinishReason(
  reason: string | undefined,
  hasToolCalls: boolean,
): string {
  if (hasToolCalls) return 'tool_calls'
  switch (reason) {
    case 'STOP':
      return 'stop'
    case 'MAX_TOKENS':
      return 'length'
    case 'SAFETY':
    case 'RECITATION':
      return 'content_filter'
    default:
      return 'stop'
  }
}

/**
 * Translate a non-streaming Gemini generateContent response to an OpenAI
 * chat-completion response.
 */
export function translateGeminiResponseToOpenAI(
  raw: GeminiGenerateContentResponse,
  model: string,
): Record<string, unknown> {
  const data = unwrap(raw)
  const candidate = data.candidates?.[0]
  const parts = candidate?.content?.parts ?? []
  const textChunks: string[] = []
  const toolCalls: Array<{
    id: string
    type: 'function'
    function: { name: string; arguments: string }
  }> = []
  for (const part of parts) {
    if (typeof part.text === 'string' && part.text) {
      textChunks.push(part.text)
    } else if (part.functionCall?.name) {
      toolCalls.push({
        id: `call_${toolCalls.length}_${Date.now().toString(36)}`,
        type: 'function',
        function: {
          name: part.functionCall.name,
          arguments: JSON.stringify(part.functionCall.args ?? {}),
        },
      })
    }
  }
  const usage = data.usageMetadata
  return {
    id: `chatcmpl_${Date.now().toString(36)}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: textChunks.join(''),
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        },
        finish_reason: mapFinishReason(
          candidate?.finishReason,
          toolCalls.length > 0,
        ),
      },
    ],
    usage: {
      prompt_tokens: usage?.promptTokenCount ?? 0,
      completion_tokens: usage?.candidatesTokenCount ?? 0,
      total_tokens:
        usage?.totalTokenCount ??
        (usage?.promptTokenCount ?? 0) + (usage?.candidatesTokenCount ?? 0),
      prompt_tokens_details: {
        cached_tokens: usage?.cachedContentTokenCount ?? 0,
      },
    },
  }
}

/**
 * Read a Code Assist SSE stream and yield OpenAI-shaped `chat.completion.chunk`
 * frames as a ReadableStream of Uint8Array — the same wire format
 * `openaiShim.ts` already knows how to parse.
 */
function makeOpenAIShimStream(
  upstream: ReadableStream<Uint8Array>,
  model: string,
): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder()
  const decoder = new TextDecoder()
  const chatId = `chatcmpl_${Date.now().toString(36)}`
  const createdAt = Math.floor(Date.now() / 1000)
  let toolCallIndex = 0
  let emittedRole = false
  let finishReason: string | undefined
  let totalUsage:
    | {
        prompt_tokens?: number
        completion_tokens?: number
        total_tokens?: number
        cached_tokens?: number
      }
    | undefined

  function encodeChunk(delta: Record<string, unknown>, extras: Record<string, unknown> = {}): Uint8Array {
    const chunk = {
      id: chatId,
      object: 'chat.completion.chunk',
      created: createdAt,
      model,
      choices: [
        {
          index: 0,
          delta,
          finish_reason: extras.finish_reason ?? null,
        },
      ],
      ...(extras.usage ? { usage: extras.usage } : {}),
    }
    return encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`)
  }

  function processPayload(
    raw: GeminiGenerateContentResponse,
    controller: ReadableStreamDefaultController<Uint8Array>,
  ): void {
    const data = unwrap(raw)
    const candidate = data.candidates?.[0]
    const parts = candidate?.content?.parts ?? []
    for (const part of parts) {
      if (typeof part.text === 'string' && part.text) {
        if (!emittedRole) {
          controller.enqueue(encodeChunk({ role: 'assistant', content: '' }))
          emittedRole = true
        }
        controller.enqueue(encodeChunk({ content: part.text }))
      } else if (part.functionCall?.name) {
        if (!emittedRole) {
          controller.enqueue(encodeChunk({ role: 'assistant', content: '' }))
          emittedRole = true
        }
        const index = toolCallIndex++
        controller.enqueue(
          encodeChunk({
            tool_calls: [
              {
                index,
                id: `call_${index}_${Date.now().toString(36)}`,
                type: 'function',
                function: {
                  name: part.functionCall.name,
                  arguments: JSON.stringify(part.functionCall.args ?? {}),
                },
              },
            ],
          }),
        )
      }
    }
    if (candidate?.finishReason) {
      finishReason = mapFinishReason(candidate.finishReason, toolCallIndex > 0)
    }
    if (data.usageMetadata) {
      totalUsage = {
        prompt_tokens: data.usageMetadata.promptTokenCount,
        completion_tokens: data.usageMetadata.candidatesTokenCount,
        total_tokens: data.usageMetadata.totalTokenCount,
        cached_tokens: data.usageMetadata.cachedContentTokenCount,
      }
    }
  }

  return new ReadableStream<Uint8Array>({
    async start(controller) {
      const reader = upstream.getReader()
      let buffer = ''
      try {
        while (true) {
          const { value, done } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          let idx: number
          while ((idx = buffer.indexOf('\n\n')) !== -1) {
            const frame = buffer.slice(0, idx)
            buffer = buffer.slice(idx + 2)
            const dataLine = frame
              .split('\n')
              .map(l => l.trim())
              .find(l => l.startsWith('data:'))
            if (!dataLine) continue
            const payload = dataLine.slice(5).trim()
            if (!payload || payload === '[DONE]') continue
            try {
              processPayload(JSON.parse(payload), controller)
            } catch {
              // Malformed frame — skip.
            }
          }
        }
        // Flush any trailing frame.
        if (buffer.trim()) {
          const dataLine = buffer
            .split('\n')
            .map(l => l.trim())
            .find(l => l.startsWith('data:'))
          if (dataLine) {
            const payload = dataLine.slice(5).trim()
            if (payload && payload !== '[DONE]') {
              try {
                processPayload(JSON.parse(payload), controller)
              } catch {
                /* ignore */
              }
            }
          }
        }
        controller.enqueue(
          encodeChunk(
            {},
            {
              finish_reason: finishReason ?? 'stop',
              ...(totalUsage
                ? {
                    usage: {
                      prompt_tokens: totalUsage.prompt_tokens ?? 0,
                      completion_tokens: totalUsage.completion_tokens ?? 0,
                      total_tokens:
                        totalUsage.total_tokens ??
                        (totalUsage.prompt_tokens ?? 0) +
                          (totalUsage.completion_tokens ?? 0),
                      prompt_tokens_details: {
                        cached_tokens: totalUsage.cached_tokens ?? 0,
                      },
                    },
                  }
                : {}),
            },
          ),
        )
        controller.enqueue(encoder.encode('data: [DONE]\n\n'))
        controller.close()
      } catch (err) {
        controller.error(err)
      }
    },
  })
}

export type GeminiCodeAssistFetchInit = {
  body: OpenAIRequestBody
  signal?: AbortSignal
  model: string
  headers?: Record<string, string>
}

/**
 * Make a Code Assist call using the current OpenAI-format request body and
 * return a `Response` whose body is OpenAI-format JSON or SSE. Meant to be
 * used as a drop-in replacement for `fetch(base + '/chat/completions', …)`
 * inside `openaiShim.ts`.
 */
export async function geminiCodeAssistFetch(
  init: GeminiCodeAssistFetchInit,
  deps: GeminiCodeAssistDeps = {},
): Promise<Response> {
  const fetchImpl = deps.fetchImpl ?? fetch
  const endpoint = deps.endpoint ?? CODE_ASSIST_ENDPOINT
  const loadToken = deps.loadToken ?? loadGeminiCliOAuthToken
  const resolveProject = deps.resolveProjectId ?? resolveGeminiCliProjectId

  let token
  try {
    token = await loadToken()
  } catch (err) {
    return new Response(
      JSON.stringify({
        error: {
          message: (err as Error).message,
          type: 'gemini_cli_oauth_error',
        },
      }),
      { status: 401, headers: { 'Content-Type': 'application/json' } },
    )
  }
  let projectId: string
  try {
    projectId = await resolveProject(token.accessToken)
  } catch (err) {
    return new Response(
      JSON.stringify({
        error: {
          message: (err as Error).message,
          type: 'gemini_code_assist_onboarding_required',
        },
      }),
      { status: 403, headers: { 'Content-Type': 'application/json' } },
    )
  }

  const geminiRequest = translateOpenAIRequestToGemini(init.body)
  const envelope: CodeAssistEnvelope = {
    model: init.model,
    project: projectId,
    request: geminiRequest,
  }

  const streaming = init.body.stream === true
  const path = streaming
    ? '/v1internal:streamGenerateContent?alt=sse'
    : '/v1internal:generateContent'
  const url = `${endpoint.replace(/\/+$/, '')}${path}`

  const response = await fetchImpl(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token.accessToken}`,
      'x-goog-api-client': 'openclaude-gemini-cli-experimental',
      ...(init.headers ?? {}),
    },
    body: JSON.stringify(envelope),
    signal: init.signal,
  })

  if (!response.ok) {
    // Re-wrap the error as an OpenAI-shaped error so the shim's existing
    // parser surfaces it cleanly. Read the upstream body here rather than
    // letting the shim double-consume the stream.
    const text = await response.text().catch(() => '')
    return new Response(
      JSON.stringify({
        error: {
          message: `Code Assist ${response.status}: ${text.slice(0, 500)}`,
          type: 'gemini_code_assist_error',
        },
      }),
      {
        status: response.status,
        headers: { 'Content-Type': 'application/json' },
      },
    )
  }

  if (!streaming) {
    const raw = (await response.json()) as GeminiGenerateContentResponse
    const openAIBody = translateGeminiResponseToOpenAI(raw, init.model)
    return new Response(JSON.stringify(openAIBody), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  if (!response.body) {
    return new Response(
      JSON.stringify({
        error: {
          message: 'Code Assist returned no stream body',
          type: 'gemini_code_assist_error',
        },
      }),
      { status: 500, headers: { 'Content-Type': 'application/json' } },
    )
  }

  const translated = makeOpenAIShimStream(response.body, init.model)
  return new Response(translated, {
    status: 200,
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    },
  })
}
