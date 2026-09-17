import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { describeError, documentsApi } from '../services/api'
import type { KnowledgeBaseInfo } from '../types'

const SELECTION_KEY = 'rd-agent-kb-selection'

function loadStoredSelection(): string[] {
  try {
    const raw = localStorage.getItem(SELECTION_KEY)
    if (!raw) return []
    const parsed: unknown = JSON.parse(raw)
    return Array.isArray(parsed)
      ? parsed.filter((item): item is string => typeof item === 'string')
      : []
  } catch {
    return []
  }
}

/**
 * The user's knowledge bases, and which of them a question may draw on.
 *
 * Selection is kept non-empty on purpose. "Nothing selected" would be ambiguous
 * — no context at all, or every base? — and the backend already reserves an
 * empty list for "search everything", so the UI never produces one: the last
 * remaining selection cannot be removed, and a base deleted elsewhere drops out
 * of the scope rather than leaving a stale id in it.
 */
export function useKnowledgeBases() {
  const [items, setItems] = useState<KnowledgeBaseInfo[]>([])
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [uploadTargetId, setUploadTargetId] = useState<string>('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const initializedRef = useRef(false)

  const refresh = useCallback(async (): Promise<KnowledgeBaseInfo[]> => {
    try {
      const list = await documentsApi.listKnowledgeBases()
      setItems(list)
      setError('')
      return list
    } catch (err) {
      setError(describeError(err))
      return []
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  // Reconcile the stored selection with what actually exists. Runs on every
  // refresh, so a base created or deleted in another tab cannot leave the scope
  // pointing at something that is gone.
  useEffect(() => {
    if (items.length === 0) return
    const ids = items.map(kb => kb.id)
    const known = new Set(ids)
    setSelectedIds(prev => {
      const source = initializedRef.current ? prev : loadStoredSelection()
      const kept = source.filter(id => known.has(id))
      return kept.length > 0 ? kept : ids
    })
    initializedRef.current = true
  }, [items])

  // The upload target is separate from the retrieval scope: a user may search
  // several bases while still filing new documents into one of them.
  useEffect(() => {
    if (items.length === 0) return
    setUploadTargetId(prev => {
      if (prev && items.some(kb => kb.id === prev)) return prev
      return items.find(kb => kb.is_default)?.id ?? items[0].id
    })
  }, [items])

  useEffect(() => {
    if (!initializedRef.current) return
    try {
      localStorage.setItem(SELECTION_KEY, JSON.stringify(selectedIds))
    } catch {
      /* ignore */
    }
  }, [selectedIds])

  const allSelected = items.length > 0 && selectedIds.length === items.length

  /** Scope to send with a question. Empty means "every base" to the backend. */
  const scopeIds = useMemo(
    () => (allSelected ? [] : selectedIds),
    [allSelected, selectedIds]
  )

  const scopeLabel = useMemo(() => {
    if (items.length === 0) return '暂无知识库'
    if (allSelected) return '全部知识库'
    const names = items.filter(kb => selectedIds.includes(kb.id)).map(kb => kb.name)
    return names.length ? names.join('、') : '全部知识库'
  }, [items, selectedIds, allSelected])

  const totalChunks = useMemo(
    () => items.reduce((sum, kb) => sum + kb.chunk_count, 0),
    [items]
  )

  const toggle = useCallback((id: string) => {
    setSelectedIds(prev => {
      if (!prev.includes(id)) return [...prev, id]
      // Refuse to empty the scope: at least one base must stay in play.
      if (prev.length <= 1) return prev
      return prev.filter(item => item !== id)
    })
  }, [])

  const selectAll = useCallback(() => {
    setSelectedIds(items.map(kb => kb.id))
  }, [items])

  const create = useCallback(
    async (name: string): Promise<string | null> => {
      try {
        const created = await documentsApi.createKnowledgeBase(name)
        await refresh()
        // Focus the new base: the next thing the user does is upload into it.
        setSelectedIds(prev => (prev.includes(created.id) ? prev : [...prev, created.id]))
        setUploadTargetId(created.id)
        return null
      } catch (err) {
        return describeError(err)
      }
    },
    [refresh]
  )

  const rename = useCallback(
    async (id: string, name: string): Promise<string | null> => {
      try {
        await documentsApi.renameKnowledgeBase(id, name)
        await refresh()
        return null
      } catch (err) {
        return describeError(err)
      }
    },
    [refresh]
  )

  const remove = useCallback(
    async (id: string): Promise<string | null> => {
      try {
        await documentsApi.deleteKnowledgeBase(id)
        await refresh()
        return null
      } catch (err) {
        return describeError(err)
      }
    },
    [refresh]
  )

  return {
    items,
    loading,
    error,
    selectedIds,
    allSelected,
    scopeIds,
    scopeLabel,
    totalChunks,
    uploadTargetId,
    setUploadTargetId,
    toggle,
    selectAll,
    create,
    rename,
    remove,
    refresh,
  }
}

/** Everything the sidebar needs, passed down as one object. */
export type KnowledgeBasesController = ReturnType<typeof useKnowledgeBases>
