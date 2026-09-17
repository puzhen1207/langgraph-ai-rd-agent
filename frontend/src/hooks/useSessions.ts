import { useCallback, useEffect, useRef, useState } from 'react'
import { v4 as uuidv4 } from 'uuid'
import { chatApi, createChatStream } from '../services/api'
import type { IntentType, Message, SessionSummary, SourceRef } from '../types'

const SESSION_KEY = 'rd-agent-session-id'

function loadStoredSessionId(): string | null {
  try {
    return localStorage.getItem(SESSION_KEY)
  } catch {
    return null
  }
}

function persistSessionId(id: string) {
  try {
    localStorage.setItem(SESSION_KEY, id)
  } catch {
    /* ignore */
  }
}

/**
 * Multi-session chat state.
 *
 * The backend keeps every conversation the user has ever started (until TTL
 * expires), and the sidebar renders them as a list. This state machine owns:
 *
 *   - the list of session summaries, ordered newest-first by the backend
 *   - which one is currently active, mirrored to localStorage so a page
 *     reload returns the user to the same conversation
 *   - the message buffer for the active session
 *   - streaming + token coalescing plumbing, kept from the original useChat
 */
export function useSessions() {
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string>(() => {
    // Defer real selection until the server list arrives; an empty string here
    // means "no conversation yet" and the UI shows a fresh empty state.
    return loadStoredSessionId() ?? ''
  })
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [currentNode, setCurrentNode] = useState<string>('')
  const abortRef = useRef<(() => void) | null>(null)

  const refreshSessions = useCallback(async () => {
    try {
      const list = (await chatApi.listSessions()) as SessionSummary[]
      setSessions(list)
      return list
    } catch {
      /* network down — keep whatever we already had */
      return [] as SessionSummary[]
    }
  }, [])

  // Pick the active session: respect localStorage if it still exists on the
  // server, otherwise fall back to the most recent conversation, otherwise
  // create one on demand.
  useEffect(() => {
    let cancelled = false
    void (async () => {
      const list = await refreshSessions()
      if (cancelled) return

      const stored = loadStoredSessionId()
      const storedKnown = stored && list.some(s => s.session_id === stored)

      if (storedKnown && stored) {
        setActiveSessionId(stored)
      } else if (list.length > 0) {
        const id = list[0].session_id
        setActiveSessionId(id)
        persistSessionId(id)
      } else {
        const id = uuidv4()
        setActiveSessionId(id)
        persistSessionId(id)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [refreshSessions])

  // Hydrate message history whenever the active session changes.
  useEffect(() => {
    if (!activeSessionId) {
      setMessages([])
      return
    }
    let cancelled = false
    chatApi
      .getHistory(activeSessionId)
      .then((res: { messages?: Array<{ role: string; content: string }> }) => {
        if (cancelled || !res?.messages?.length) {
          if (!cancelled) setMessages([])
          return
        }
        setMessages(
          res.messages.map((m) => ({
            id: uuidv4(),
            role: m.role as 'user' | 'assistant',
            content: m.content,
            timestamp: new Date(),
          }))
        )
      })
      .catch(() => {
        if (!cancelled) setMessages([])
      })
    return () => {
      cancelled = true
    }
  }, [activeSessionId])

  const appendMessage = useCallback((msg: Message) => {
    setMessages(prev => [...prev, msg])
  }, [])

  const updateLastAssistant = useCallback((updater: (m: Message) => Message) => {
    setMessages(prev => {
      const idx = [...prev].reverse().findIndex(m => m.role === 'assistant')
      if (idx === -1) return prev
      const realIdx = prev.length - 1 - idx
      const updated = [...prev]
      updated[realIdx] = updater(updated[realIdx])
      return updated
    })
  }, [])

  const pendingTokensRef = useRef('')
  const flushFrameRef = useRef<number | null>(null)

  const cancelPendingFlush = useCallback(() => {
    if (flushFrameRef.current !== null) {
      cancelAnimationFrame(flushFrameRef.current)
      flushFrameRef.current = null
    }
  }, [])

  const flushTokens = useCallback(() => {
    cancelPendingFlush()
    const pending = pendingTokensRef.current
    if (!pending) return
    pendingTokensRef.current = ''
    updateLastAssistant(m => ({ ...m, content: m.content + pending }))
  }, [cancelPendingFlush, updateLastAssistant])

  const dropTokens = useCallback(() => {
    cancelPendingFlush()
    pendingTokensRef.current = ''
  }, [cancelPendingFlush])

  const bufferToken = useCallback(
    (token: string) => {
      pendingTokensRef.current += token
      if (flushFrameRef.current !== null) return
      flushFrameRef.current = requestAnimationFrame(() => {
        flushFrameRef.current = null
        flushTokens()
      })
    },
    [flushTokens]
  )

  useEffect(() => cancelPendingFlush, [cancelPendingFlush])

  const sendMessage = useCallback(
    async (query: string, useStream = true, kbIds: string[] = []) => {
      if (!query.trim() || isLoading) return

      const userMsg: Message = {
        id: uuidv4(),
        role: 'user',
        content: query,
        timestamp: new Date(),
      }
      appendMessage(userMsg)
      setIsLoading(true)
      setCurrentNode('query_analyze')

      if (useStream) {
        const streamingId = uuidv4()
        const assistantMsg: Message = {
          id: streamingId,
          role: 'assistant',
          content: '',
          timestamp: new Date(),
          isStreaming: true,
        }
        appendMessage(assistantMsg)

        abortRef.current = createChatStream(
          { query, session_id: activeSessionId, knowledge_base_ids: kbIds },
          {
            onSession: (sid) => {
              setActiveSessionId(sid)
              persistSessionId(sid)
            },
            onProgress: (node, extra) => {
              setCurrentNode(node)
              const intent = extra?.intent as IntentType | undefined
              const retrieved = extra?.retrieved as number | undefined
              const sources = extra?.sources as SourceRef[] | undefined
              if (intent || retrieved !== undefined || sources) {
                updateLastAssistant(m => ({
                  ...m,
                  intent: intent ?? m.intent,
                  retrievedCount: retrieved ?? m.retrievedCount,
                  sources: sources ?? m.sources,
                }))
              }
            },
            onGenerationStart: () => {
              setCurrentNode('llm_generate')
              dropTokens()
              updateLastAssistant(m => ({ ...m, content: '' }))
            },
            onToken: (token) => bufferToken(token),
            onDone: (metadata) => {
              flushTokens()
              updateLastAssistant(m => ({
                ...m,
                isStreaming: false,
                intent: (metadata?.intent as IntentType | undefined) ?? m.intent,
                retrievedCount:
                  (metadata?.retrieved_count as number | undefined) ?? m.retrievedCount,
                sources: (metadata?.sources as SourceRef[] | undefined) ?? m.sources,
                isValid: metadata?.is_valid as boolean | undefined,
                validationReason: metadata?.validation_reason as string | undefined,
              }))
              setIsLoading(false)
              setCurrentNode('')
              abortRef.current = null
              // The server just bumped updated_at (and may have seeded a
              // title from this turn) — pull the fresh list so the sidebar
              // ordering reflects reality.
              void refreshSessions()
            },
            onError: (err) => {
              flushTokens()
              updateLastAssistant(m => ({
                ...m,
                content: m.content || `错误：${err}`,
                isStreaming: false,
              }))
              setIsLoading(false)
              setCurrentNode('')
              abortRef.current = null
            },
          }
        )
      } else {
        try {
          const res = await chatApi.send({
            query,
            session_id: activeSessionId,
            knowledge_base_ids: kbIds,
          })
          appendMessage({
            id: uuidv4(),
            role: 'assistant',
            content: res.response,
            intent: res.intent,
            timestamp: new Date(),
            retrievedCount: res.retrieved_count,
            sources: res.sources,
            isValid: res.is_valid,
            validationReason: res.validation_reason,
          })
          void refreshSessions()
        } catch (err) {
          appendMessage({
            id: uuidv4(),
            role: 'assistant',
            content: `请求失败：${String(err)}`,
            timestamp: new Date(),
          })
        } finally {
          setIsLoading(false)
          setCurrentNode('')
        }
      }
    },
    [
      isLoading,
      activeSessionId,
      appendMessage,
      updateLastAssistant,
      bufferToken,
      dropTokens,
      flushTokens,
      refreshSessions,
    ]
  )

  /**
   * Start a brand-new conversation while keeping the old one in the list.
   *
   * "Clear" was the old verb; the action has always been "abandon current and
   * start fresh". Naming it more honestly matches what the sidebar now shows.
   */
  const createSession = useCallback(() => {
    abortRef.current?.()
    dropTokens()
    const id = uuidv4()
    setActiveSessionId(id)
    persistSessionId(id)
    setMessages([])
    setIsLoading(false)
    setCurrentNode('')
  }, [dropTokens])

  /** Switch the active session and replace the message buffer with its history. */
  const switchSession = useCallback((id: string) => {
    if (!id) return
    abortRef.current?.()
    dropTokens()
    setActiveSessionId(id)
    persistSessionId(id)
    setIsLoading(false)
    setCurrentNode('')
  }, [dropTokens])

  const stopStreaming = useCallback(() => {
    abortRef.current?.()
    flushTokens()
    updateLastAssistant(m => ({ ...m, isStreaming: false }))
    setIsLoading(false)
    setCurrentNode('')
    abortRef.current = null
  }, [updateLastAssistant, flushTokens])

  return {
    sessions,
    activeSessionId,
    messages,
    isLoading,
    currentNode,
    sendMessage,
    createSession,
    switchSession,
    stopStreaming,
    refreshSessions,
  }
}