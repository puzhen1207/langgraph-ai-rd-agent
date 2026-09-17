import axios from 'axios'
import { v4 as uuidv4 } from 'uuid'
import type {
  ChatRequest,
  ChatResponse,
  ChunkItem,
  DocumentStats,
  HealthResponse,
  KnowledgeBaseInfo,
  UploadResponse,
} from '../types'

const BASE = '/api/v1'
const ADMIN_KEY_STORAGE = 'rd-agent-admin-key'
const CLIENT_TOKEN_STORAGE = 'rd-agent-client-token'

/**
 * Anonymous identity for this browser.
 *
 * The backend binds every session to the token that created it, so knowing a
 * session id alone is no longer enough to read or delete that conversation.
 */
function loadClientToken(): string {
  try {
    const saved = localStorage.getItem(CLIENT_TOKEN_STORAGE)
    if (saved) return saved
  } catch {
    /* storage unavailable (private mode); fall through to an ephemeral token */
  }
  const token = uuidv4()
  try {
    localStorage.setItem(CLIENT_TOKEN_STORAGE, token)
  } catch {
    /* ignore */
  }
  return token
}

const clientToken = loadClientToken()

const http = axios.create({
  baseURL: BASE,
  timeout: 120_000,
  headers: { 'X-Client-Token': clientToken },
})
const uploadHttp = axios.create({
  baseURL: BASE,
  timeout: 600_000,
  headers: { 'X-Client-Token': clientToken },
})

function adminHeaders(): Record<string, string> {
  const key = sessionStorage.getItem(ADMIN_KEY_STORAGE)
  return key ? { 'X-Admin-Key': key } : {}
}

export function setAdminKey(key: string) {
  if (key.trim()) sessionStorage.setItem(ADMIN_KEY_STORAGE, key.trim())
  else sessionStorage.removeItem(ADMIN_KEY_STORAGE)
}

/**
 * Turn an axios failure into something worth showing a user.
 *
 * FastAPI puts a human-readable reason in `detail`, and for validation errors
 * that detail is a list of objects. Rendering `String(err)` instead produced
 * "AxiosError: Request failed with status code 409" — technically true, and
 * useless for the one case that matters most here, a duplicate base name.
 */
export function describeError(err: unknown): string {
  const detail = (err as { response?: { data?: { detail?: unknown } } })?.response
    ?.data?.detail
  if (typeof detail === 'string' && detail.trim()) return detail
  if (Array.isArray(detail)) {
    const messages = detail
      .map(item => (item as { msg?: string })?.msg)
      .filter((msg): msg is string => Boolean(msg))
    if (messages.length) return messages.join('；')
  }
  return String(err)
}

export const chatApi = {
  send: (req: ChatRequest): Promise<ChatResponse> =>
    http.post<ChatResponse>('/chat', req).then(r => r.data),
  clearSession: (sessionId: string) =>
    http.delete(`/chat/session/${sessionId}`).then(r => r.data),
  getHistory: (sessionId: string) =>
    http.get(`/chat/session/${sessionId}/history`).then(r => r.data),
  /** Every conversation this browser has ever started, newest first. */
  listSessions: () =>
    http.get('/chat/sessions').then(r => r.data),
}

export const documentsApi = {
  listKnowledgeBases: (): Promise<KnowledgeBaseInfo[]> =>
    http.get<KnowledgeBaseInfo[]>('/documents/knowledge-bases').then(r => r.data),

  createKnowledgeBase: (name: string, description = ''): Promise<KnowledgeBaseInfo> =>
    http
      .post<KnowledgeBaseInfo>(
        '/documents/knowledge-bases',
        { name, description },
        { headers: adminHeaders() }
      )
      .then(r => r.data),

  renameKnowledgeBase: (id: string, name: string): Promise<KnowledgeBaseInfo> =>
    http
      .patch<KnowledgeBaseInfo>(
        `/documents/knowledge-bases/${id}`,
        { name },
        { headers: adminHeaders() }
      )
      .then(r => r.data),

  deleteKnowledgeBase: (id: string) =>
    http
      .delete(`/documents/knowledge-bases/${id}`, { headers: adminHeaders() })
      .then(r => r.data),

  /** Remove every passage in one base, keeping the base itself. */
  clearKnowledgeBase: (id: string) =>
    http
      .delete(`/documents/knowledge-bases/${id}/documents`, { headers: adminHeaders() })
      .then(r => r.data),

  upload: (file: File, knowledgeBaseId: string): Promise<UploadResponse> => {
    const form = new FormData()
    form.append('file', file)
    form.append('knowledge_base_id', knowledgeBaseId)
    return uploadHttp
      .post<UploadResponse>('/documents/upload', form, {
        headers: { 'Content-Type': 'multipart/form-data', ...adminHeaders() },
      })
      .then(r => r.data)
  },

  uploadText: (
    content: string,
    source: string,
    knowledgeBaseId: string,
    docType = 'markdown'
  ): Promise<UploadResponse> => {
    const form = new FormData()
    form.append('content', content)
    form.append('source', source)
    form.append('knowledge_base_id', knowledgeBaseId)
    form.append('doc_type', docType)
    return http
      .post<UploadResponse>('/documents/upload-text', form, { headers: adminHeaders() })
      .then(r => r.data)
  },

  getStats: (knowledgeBaseId?: string): Promise<DocumentStats> =>
    http
      .get<DocumentStats>('/documents/stats', {
        params: knowledgeBaseId ? { knowledge_base_id: knowledgeBaseId } : undefined,
      })
      .then(r => r.data),

  clear: () =>
    http.delete('/documents/clear', { headers: adminHeaders() }).then(r => r.data),

  getChunks: (knowledgeBaseId?: string): Promise<ChunkItem[]> =>
    http
      .get<ChunkItem[]>('/documents/chunks', {
        headers: adminHeaders(),
        params: knowledgeBaseId ? { knowledge_base_id: knowledgeBaseId } : undefined,
      })
      .then(r => r.data),
}

export const healthApi = {
  check: (): Promise<HealthResponse> => axios.get<HealthResponse>('/health').then(r => r.data),
}

export function createChatStream(
  req: ChatRequest,
  callbacks: {
    onSession?: (sessionId: string) => void
    onProgress?: (node: string, extra?: Record<string, unknown>) => void
    onGenerationStart?: () => void
    onToken?: (token: string) => void
    onDone?: (metadata?: Record<string, unknown>) => void
    onError?: (msg: string) => void
  }
): () => void {
  let buffer = ''
  let finished = false
  const controller = new AbortController()

  void (async () => {
    try {
      const res = await fetch(`${BASE}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Client-Token': clientToken,
        },
        body: JSON.stringify(req),
        signal: controller.signal,
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      if (!res.body) throw new Error('Streaming response has no body')

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      while (!finished) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const data = JSON.parse(line.slice(6))
            if (data.type === 'session') callbacks.onSession?.(data.session_id)
            else if (data.type === 'progress') callbacks.onProgress?.(data.node, data)
            else if (data.type === 'generation_start') callbacks.onGenerationStart?.()
            else if (data.type === 'token') callbacks.onToken?.(data.content)
            else if (data.type === 'done') {
              finished = true
              callbacks.onDone?.(data)
              await reader.cancel()
              break
            } else if (data.type === 'error') {
              finished = true
              callbacks.onError?.(data.message)
              await reader.cancel()
              break
            }
          } catch { /* ignore malformed individual events */ }
        }
      }

      if (!finished) {
        finished = true
        callbacks.onError?.('流式连接意外中断')
      }
    } catch (err) {
      if (!finished && (err as Error).name !== 'AbortError') {
        finished = true
        callbacks.onError?.(String(err))
      }
    }
  })()

  return () => {
    finished = true
    controller.abort()
  }
}
