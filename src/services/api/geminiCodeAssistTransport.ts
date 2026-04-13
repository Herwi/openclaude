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

function newToolCallId(): string {
  // crypto.randomUUID is available in both Node 18+ and Bun.
  const id =
    typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID().replace(/-/g, '')
      : Math.random().toString(36).slice(2)
  return `call_${id}`
}

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
    extra_content?: Record<string, unknown>
  }>
  tool_call_id?: string
  name?: string
}

// Magic value used by openaiShim's convertMessages to "ask" Google's
// OpenAI-compat layer to skip thought-signature validation. It does not have
// any meaning for the native Code Assist endpoint; we must NOT forward it as
// a real signature, otherwise the server will reject it as malformed.
const SKIP_THOUGHT_SIGNATURE_PLACEHOLDER = 'skip_thought_signature_validator'

// Code Assist enforces `1 <= maxOutputTokens < 65536`. Claude Code defaults
// `max_tokens` to exactly 65536 for Sonnet-tier models, which sits on the
// exclusive upper bound and produces a 400 INVALID_ARGUMENT. Clamp into the
// server's accepted range before sending.
const CODE_ASSIST_MAX_OUTPUT_TOKENS = 65535

function clampMaxOutputTokens(value: number): number {
  if (!Number.isFinite(value)) return CODE_ASSIST_MAX_OUTPUT_TOKENS
  if (value < 1) return 1
  if (value > CODE_ASSIST_MAX_OUTPUT_TOKENS) return CODE_ASSIST_MAX_OUTPUT_TOKENS
  return Math.floor(value)
}

function extractThoughtSignature(
  extraContent: Record<string, unknown> | undefined,
): string | undefined {
  if (!extraContent) return undefined
  const google = extraContent.google as Record<string, unknown> | undefined
  const value = google?.thought_signature
  if (typeof value !== 'string') return undefined
  if (!value || value === SKIP_THOUGHT_SIGNATURE_PLACEHOLDER) return undefined
  return value
}

/**
 * Code Assist's `v1internal:generateContent` validates tool schemas against a
 * strict OpenAPI 3.0 subset. It is dramatically stricter than Google's
 * OpenAI-compat layer and stricter than `normalizeSchemaForOpenAI` cleans up
 * for. Unknown JSON Schema keywords produce 400 INVALID_ARGUMENT, killing the
 * request before it ever reaches the model. This sanitizer uses an allow-list
 * of keys the server accepts and drops everything else.
 *
 * Transformations applied (beyond stripping):
 *   - `const: X` is rewritten to `enum: [X]` because Gemini accepts enum but
 *     has no `const` keyword.
 *   - `oneOf` is rewritten to `anyOf` because Gemini only implements anyOf.
 *     This is a small semantic loosening (exclusive-or → inclusive-or) but
 *     is what the generative Gemini API itself expects.
 *   - `allOf` is dropped (merging is non-trivial and Gemini can't verify it).
 *   - `type: 'null'` or `type: ['string', 'null']` is rewritten to
 *     `nullable: true` + single concrete type.
 *   - `required` is filtered to property names that survived sanitization, so
 *     we never ask for a field that no longer exists.
 */
const CODE_ASSIST_NUMERIC_KEYS = [
  'minimum',
  'maximum',
  'minLength',
  'maxLength',
  'minItems',
  'maxItems',
  'minProperties',
  'maxProperties',
  'pattern',
] as const

export function sanitizeSchemaForCodeAssist(input: unknown): Record<string, unknown> {
  if (Array.isArray(input) || !input || typeof input !== 'object') {
    return {}
  }
  const schema = input as Record<string, unknown>
  const out: Record<string, unknown> = {}

  // ---- type + nullable ----
  const typeValue = schema.type
  let nullable = false
  if (Array.isArray(typeValue)) {
    const nonNull: string[] = []
    for (const t of typeValue) {
      if (t === 'null') nullable = true
      else if (typeof t === 'string') nonNull.push(t)
    }
    // Gemini's single-type field cannot express a union; pick the first
    // concrete type. This loses information but is the closest thing
    // representable.
    if (nonNull.length >= 1) out.type = nonNull[0]
  } else if (typeValue === 'null') {
    nullable = true
  } else if (typeof typeValue === 'string') {
    out.type = typeValue
  }
  if (nullable) out.nullable = true
  if (schema.nullable === true) out.nullable = true

  // ---- description / title ----
  if (typeof schema.description === 'string') out.description = schema.description
  if (typeof schema.title === 'string') out.title = schema.title

  // ---- enum / const ----
  // A real enum on the source wins over a const. Otherwise promote const.
  if (Array.isArray(schema.enum) && schema.enum.length > 0) {
    out.enum = [...schema.enum]
  } else if ('const' in schema) {
    out.enum = [schema.const]
  }

  // ---- properties (recurse) ----
  if (
    schema.properties &&
    typeof schema.properties === 'object' &&
    !Array.isArray(schema.properties)
  ) {
    const props: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(schema.properties)) {
      props[k] = sanitizeSchemaForCodeAssist(v)
    }
    out.properties = props
  }

  // ---- items (recurse) ----
  if ('items' in schema && schema.items != null) {
    out.items = Array.isArray(schema.items)
      ? schema.items.map(sanitizeSchemaForCodeAssist)
      : sanitizeSchemaForCodeAssist(schema.items)
  }

  // ---- anyOf / oneOf (drop allOf) ----
  if (Array.isArray(schema.anyOf) && schema.anyOf.length > 0) {
    out.anyOf = schema.anyOf.map(sanitizeSchemaForCodeAssist)
  } else if (Array.isArray(schema.oneOf) && schema.oneOf.length > 0) {
    out.anyOf = schema.oneOf.map(sanitizeSchemaForCodeAssist)
  }

  // ---- numeric / string / array constraints (pass-through) ----
  for (const key of CODE_ASSIST_NUMERIC_KEYS) {
    if (key in schema) out[key] = schema[key]
  }

  // ---- required (filter to surviving property names) ----
  if (Array.isArray(schema.required)) {
    const propNames =
      out.properties && typeof out.properties === 'object'
        ? Object.keys(out.properties as Record<string, unknown>)
        : null
    const filtered = (schema.required as unknown[]).filter(
      (r): r is string =>
        typeof r === 'string' &&
        (propNames === null || propNames.includes(r)),
    )
    if (filtered.length > 0) out.required = filtered
  }

  return out
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
  | {
      functionCall: { name: string; args: Record<string, unknown> }
      // Gemini requires this echoed back on every replayed function call so it
      // can prove the call really came from a previous model turn (and wasn't
      // forged by the client). Code Assist returns it on response parts and
      // expects the same value on the corresponding part in the next turn's
      // contents.
      thoughtSignature?: string
    }
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
  // OpenAI's tool-role messages only carry `tool_call_id`. Build a map as we
  // process assistant turns so that a later 'tool' role message can resolve
  // its function name.
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
          const signature = extractThoughtSignature(tc.extra_content)
          parts.push({
            functionCall: {
              name: tc.function.name,
              args: stringToRecord(tc.function.arguments),
            },
            ...(signature ? { thoughtSignature: signature } : {}),
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
          if (t.function.parameters) {
            // Code Assist is significantly stricter than Google's OpenAI-compat
            // layer: it rejects JSON Schema keywords Claude Code's tool schemas
            // use (const, exclusiveMinimum, oneOf, additionalProperties, …).
            // Run every tool parameter schema through our Gemini-native
            // sanitizer here — not in openaiShim — so the transport boundary
            // is the single source of truth for "what Code Assist will accept".
            decl.parameters = sanitizeSchemaForCodeAssist(t.function.parameters)
          }
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
  if (typeof maxTokens === 'number') {
    generationConfig.maxOutputTokens = clampMaxOutputTokens(maxTokens)
  }
  if (Object.keys(generationConfig).length > 0) {
    request.generationConfig = generationConfig
  }

  return request
}

type GeminiResponsePart = {
  text?: string
  functionCall?: { name?: string; args?: Record<string, unknown> }
  // Gemini stamps replay-validation tokens on parts that came from the model
  // turn. We must echo the same value back on the corresponding part in the
  // next turn's contents, otherwise the server rejects the replay.
  thoughtSignature?: string
}

type GeminiCandidate = {
  content?: {
    role?: string
    parts?: GeminiResponsePart[]
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
    extra_content?: Record<string, unknown>
  }> = []
  for (const part of parts) {
    if (typeof part.text === 'string' && part.text) {
      textChunks.push(part.text)
    } else if (part.functionCall?.name) {
      // Surface Gemini's thoughtSignature back through extra_content.google so
      // openaiShim's _convertNonStreamingResponse promotes it to the Anthropic
      // tool_use block as `signature`. Without this round-trip Code Assist
      // rejects the next turn's replayed function call.
      const extraContent =
        typeof part.thoughtSignature === 'string' && part.thoughtSignature
          ? { google: { thought_signature: part.thoughtSignature } }
          : undefined
      toolCalls.push({
        id: newToolCallId(),
        type: 'function',
        function: {
          name: part.functionCall.name,
          arguments: JSON.stringify(part.functionCall.args ?? {}),
        },
        ...(extraContent ? { extra_content: extraContent } : {}),
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
        const extraContent =
          typeof part.thoughtSignature === 'string' && part.thoughtSignature
            ? { google: { thought_signature: part.thoughtSignature } }
            : undefined
        controller.enqueue(
          encodeChunk({
            tool_calls: [
              {
                index,
                id: newToolCallId(),
                type: 'function',
                function: {
                  name: part.functionCall.name,
                  arguments: JSON.stringify(part.functionCall.args ?? {}),
                },
                ...(extraContent ? { extra_content: extraContent } : {}),
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
      // SSE frames are delimited by a blank line. The spec allows LF, CR, or
      // CRLF line endings, but Google Code Assist's `streamGenerateContent`
      // endpoint uses CRLF (`\r\n\r\n` between events) in practice. Normalize
      // CRLF to LF as bytes come in so we can frame on `\n\n` consistently.
      // An earlier version of this code only looked for `\n\n` and never
      // found a single frame — every streaming response from Code Assist was
      // truncated to whatever the (broken) flush path happened to salvage.
      let buffer = ''

      function processFrame(frame: string): void {
        // A single SSE event may contain multiple `data:` lines; per the spec
        // they are concatenated with `\n`. Code Assist only ever sends one
        // `data:` line per event today, but we handle the general case.
        const dataLines: string[] = []
        for (const rawLine of frame.split('\n')) {
          const line = rawLine.trim()
          if (line.startsWith('data:')) {
            dataLines.push(line.slice(5).trimStart())
          }
        }
        if (dataLines.length === 0) return
        const payload = dataLines.join('\n').trim()
        if (!payload || payload === '[DONE]') return
        try {
          processPayload(JSON.parse(payload), controller)
        } catch {
          // Malformed frame — skip.
        }
      }

      try {
        while (true) {
          const { value, done } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')
          let idx: number
          while ((idx = buffer.indexOf('\n\n')) !== -1) {
            const frame = buffer.slice(0, idx)
            buffer = buffer.slice(idx + 2)
            processFrame(frame)
          }
        }
        // Flush any trailing frame that didn't end with a blank line.
        if (buffer.trim().length > 0) {
          processFrame(buffer)
          buffer = ''
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

function errorResponse(
  status: number,
  message: string,
  type: string,
): Response {
  return new Response(
    JSON.stringify({ error: { message, type } }),
    { status, headers: { 'Content-Type': 'application/json' } },
  )
}

/**
 * Make a Code Assist call using the current OpenAI-format request body and
 * return a `Response` whose body is OpenAI-format JSON or SSE. Meant to be
 * used as a drop-in replacement for `fetch(base + '/chat/completions', …)`
 * inside `openaiShim.ts`.
 *
 * On a 401 from Code Assist, the OAuth token is force-refreshed and the
 * request is retried once. This handles the case where the cached
 * `expiry_date` is still "valid" in local time but the server has already
 * revoked the token (clock skew, rotation, manual revocation).
 */
export async function geminiCodeAssistFetch(
  init: GeminiCodeAssistFetchInit,
  deps: GeminiCodeAssistDeps = {},
): Promise<Response> {
  const fetchImpl = deps.fetchImpl ?? fetch
  const endpoint = deps.endpoint ?? CODE_ASSIST_ENDPOINT
  const loadToken = deps.loadToken ?? loadGeminiCliOAuthToken
  const resolveProject = deps.resolveProjectId ?? resolveGeminiCliProjectId

  // Validate the request body at the boundary — the rest of this function
  // trusts the OpenAI shape, and an upstream refactor that forgets to build
  // `messages` would otherwise crash in translateOpenAIRequestToGemini with
  // a confusing NPE instead of a clear error the user can act on.
  if (!init.body || !Array.isArray(init.body.messages)) {
    return errorResponse(
      400,
      'geminiCodeAssistFetch: request body is missing `messages[]`. This is a bug in openaiShim wiring.',
      'gemini_code_assist_bad_request',
    )
  }
  if (!init.model) {
    return errorResponse(
      400,
      'geminiCodeAssistFetch: `model` is required.',
      'gemini_code_assist_bad_request',
    )
  }

  const geminiRequest = translateOpenAIRequestToGemini(init.body)
  const streaming = init.body.stream === true
  const path = streaming
    ? '/v1internal:streamGenerateContent?alt=sse'
    : '/v1internal:generateContent'
  const url = `${endpoint.replace(/\/+$/, '')}${path}`

  async function postOnce(
    forceRefresh: boolean,
  ): Promise<{ response: Response; accessToken: string }> {
    const token = await loadToken({ forceRefresh })
    const projectId = await resolveProject(token.accessToken)
    const envelope: CodeAssistEnvelope = {
      model: init.model,
      project: projectId,
      request: geminiRequest,
    }
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
    return { response, accessToken: token.accessToken }
  }

  let response: Response
  try {
    ;({ response } = await postOnce(false))
  } catch (err) {
    // loadToken / resolveProject failures land here. Differentiate between:
    //  - onboarding / missing-project errors (403) — user should run gemini
    //    once and accept Code Assist terms
    //  - loadCodeAssist client-side errors like a 400 INVALID_ARGUMENT (400)
    //    — these are bugs on our side, not something the user can fix by
    //    re-authenticating
    //  - OAuth loader errors (401) — stale / missing / unreadable creds
    const message = (err as Error).message ?? 'unknown error'
    if (/cloudaicompanionProject/i.test(message)) {
      return errorResponse(403, message, 'gemini_code_assist_onboarding_required')
    }
    const loadCodeAssistMatch = message.match(
      /Code Assist loadCodeAssist failed \((\d{3})\)/i,
    )
    if (loadCodeAssistMatch) {
      const upstreamStatus = Number(loadCodeAssistMatch[1])
      const type =
        upstreamStatus === 400
          ? 'gemini_code_assist_bad_request'
          : 'gemini_code_assist_onboarding_required'
      return errorResponse(upstreamStatus, message, type)
    }
    return errorResponse(401, message, 'gemini_cli_oauth_error')
  }

  // 401 → cached token is dead on the server. Force a refresh and retry once.
  if (response.status === 401) {
    // Drain the original body so the underlying connection can be reused.
    await response.text().catch(() => {})
    try {
      ;({ response } = await postOnce(true))
    } catch (err) {
      return errorResponse(
        401,
        `Gemini CLI OAuth token refresh failed after 401: ${(err as Error).message}`,
        'gemini_cli_oauth_error',
      )
    }
  }

  if (!response.ok) {
    // Re-wrap upstream errors as OpenAI-shaped errors. Read the body here
    // rather than letting the shim double-consume the stream.
    const text = await response.text().catch(() => '')
    return errorResponse(
      response.status,
      `Code Assist ${response.status}: ${text.slice(0, 500)}`,
      'gemini_code_assist_error',
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
    return errorResponse(
      500,
      'Code Assist returned no stream body',
      'gemini_code_assist_error',
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
