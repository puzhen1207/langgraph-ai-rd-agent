import { useState } from 'react'
import { Check, Pencil, Plus, Trash2 } from 'lucide-react'
import Dialog from './Dialog'
import type { KnowledgeBaseInfo } from '../types'

interface Props {
  items: KnowledgeBaseInfo[]
  selectedIds: string[]
  loading: boolean
  onToggle: (id: string) => void
  onSelectAll: () => void
  onCreate: (name: string) => Promise<string | null>
  onRename: (id: string, name: string) => Promise<string | null>
  onDelete: (id: string) => Promise<string | null>
}

/**
 * Create, name, scope and delete knowledge bases.
 *
 * The checkbox is the retrieval scope, not simply a highlight: an unchecked base
 * is excluded from every future answer and from the citations shown with it.
 */
export default function KnowledgeBasePanel({
  items,
  selectedIds,
  loading,
  onToggle,
  onSelectAll,
  onCreate,
  onRename,
  onDelete,
}: Props) {
  const [creating, setCreating] = useState(false)
  const [draft, setDraft] = useState('')
  const [renamingId, setRenamingId] = useState<string | null>(null)
  const [renameDraft, setRenameDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [formError, setFormError] = useState('')
  const [deleteTarget, setDeleteTarget] = useState<KnowledgeBaseInfo | null>(null)

  const submitCreate = async (event: React.FormEvent) => {
    event.preventDefault()
    const name = draft.trim()
    if (!name || busy) return
    setBusy(true)
    const error = await onCreate(name)
    setBusy(false)
    if (error) {
      setFormError(error)
      return
    }
    setCreating(false)
    setDraft('')
    setFormError('')
  }

  const submitRename = async (id: string) => {
    const name = renameDraft.trim()
    if (!name || busy) {
      setRenamingId(null)
      return
    }
    setBusy(true)
    const error = await onRename(id, name)
    setBusy(false)
    if (error) {
      setFormError(error)
      return
    }
    setRenamingId(null)
    setFormError('')
  }

  const handleDelete = (kb: KnowledgeBaseInfo) => setDeleteTarget(kb)

  const confirmDelete = async () => {
    if (!deleteTarget || busy) return
    setBusy(true)
    const error = await onDelete(deleteTarget.id)
    setBusy(false)
    setFormError(error ?? '')
    setDeleteTarget(null)
  }

  return (
    <div className="border-t border-paper-300 px-4 py-4">
      <div className="mb-2.5 flex items-center justify-between">
        <p className="text-[11px] font-medium tracking-[0.2em] text-ink-600">
          知识库
        </p>
        {items.length > 0 && selectedIds.length < items.length && (
          <button
            onClick={onSelectAll}
            className="text-[11px] text-ink-600 transition-colors hover:text-brand-600"
          >
            全选
          </button>
        )}
      </div>

      {loading && items.length === 0 ? (
        <p className="py-1.5 text-[11px] text-ink-500">加载中…</p>
      ) : (
        <div className="space-y-0.5">
          {items.map(kb => {
            const checked = selectedIds.includes(kb.id)
            const isRenaming = renamingId === kb.id

            if (isRenaming) {
              return (
                <div key={kb.id} className="flex items-center gap-1.5 px-2 py-1">
                  <input
                    autoFocus
                    value={renameDraft}
                    onChange={e => setRenameDraft(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter') void submitRename(kb.id)
                      if (e.key === 'Escape') setRenamingId(null)
                    }}
                    className="min-w-0 flex-1 rounded-md border border-brand-300 bg-paper-50 px-1.5 py-1 text-[12px] text-ink-800 outline-none focus:border-brand-500"
                  />
                  <button
                    onClick={() => void submitRename(kb.id)}
                    className="text-[11px] text-brand-600 hover:text-brand-700"
                  >
                    保存
                  </button>
                </div>
              )
            }

            return (
              <div
                key={kb.id}
                className="group flex items-start gap-2 rounded-lg px-2 py-1.5 transition-colors hover:bg-paper-200/60"
              >
                <button
                  onClick={() => onToggle(kb.id)}
                  role="checkbox"
                  aria-checked={checked}
                  aria-label={`${checked ? '移出' : '加入'}检索范围：${kb.name}`}
                  className={`mt-0.5 flex h-3.5 w-3.5 flex-shrink-0 items-center justify-center rounded border transition-colors ${
                    checked
                      ? 'border-brand-600 bg-brand-600 text-paper-50'
                      : 'border-ink-400 bg-paper-50'
                  }`}
                >
                  {checked && <Check size={10} strokeWidth={3} />}
                </button>

                <button
                  onClick={() => onToggle(kb.id)}
                  className="min-w-0 flex-1 text-left"
                  title={kb.description || kb.name}
                >
                  <span className="flex items-center gap-1.5">
                    <span className="truncate text-[12px] font-medium text-ink-700">
                      {kb.name}
                    </span>
                    {kb.is_default && (
                      <span className="flex-shrink-0 rounded border border-paper-300 px-1 text-[10px] text-ink-600">
                        内置
                      </span>
                    )}
                  </span>
                  <span className="mt-0.5 block text-[11px] text-ink-600">
                    {kb.document_count} 篇 · {kb.chunk_count} 段
                  </span>
                </button>

                <div className="flex flex-shrink-0 items-center gap-1 pt-0.5 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
                  <button
                    onClick={() => {
                      setRenamingId(kb.id)
                      setRenameDraft(kb.name)
                      setFormError('')
                    }}
                    title="重命名"
                    className="text-ink-500 transition-colors hover:text-brand-600"
                  >
                    <Pencil size={11} />
                  </button>
                  {!kb.is_default && (
                    <button
                      onClick={() => void handleDelete(kb)}
                      title="删除知识库"
                      className="text-ink-500 transition-colors hover:text-red-500"
                    >
                      <Trash2 size={11} />
                    </button>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {creating ? (
        <form onSubmit={submitCreate} className="mt-2">
          <input
            autoFocus
            value={draft}
            onChange={e => setDraft(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Escape') {
                setCreating(false)
                setDraft('')
                setFormError('')
              }
            }}
            placeholder="知识库名称，如：深度学习文献"
            maxLength={60}
            className="w-full rounded-md border border-brand-300 bg-paper-50 px-2 py-1.5 text-[12px] text-ink-800 outline-none placeholder:text-ink-400 focus:border-brand-500"
          />
          <div className="mt-1.5 flex items-center gap-2">
            <button
              type="submit"
              disabled={!draft.trim() || busy}
              className="rounded-md bg-brand-700 px-2.5 py-1 text-[11px] text-paper-50 transition-colors hover:bg-brand-800 disabled:cursor-not-allowed disabled:bg-paper-300 disabled:text-ink-400"
            >
              创建
            </button>
            <button
              type="button"
              onClick={() => {
                setCreating(false)
                setDraft('')
                setFormError('')
              }}
              className="text-[11px] text-ink-600 transition-colors hover:text-ink-800"
            >
              取消
            </button>
          </div>
        </form>
      ) : (
        <button
          onClick={() => {
            setCreating(true)
            setFormError('')
          }}
          className="mt-2 flex w-full items-center justify-center gap-1.5 rounded-md border border-dashed border-paper-300 py-1.5 text-[11px] text-ink-500 transition-colors hover:border-brand-200 hover:text-brand-600"
        >
          <Plus size={12} strokeWidth={1.75} /> 新建知识库
        </button>
      )}

      {formError && (
        <p className="mt-1.5 rounded-md bg-red-50 px-2 py-1.5 text-[11px] leading-relaxed text-red-600 dark:bg-red-950/40 dark:text-red-400">
          {formError}
        </p>
      )}

      <p className="mt-2 text-[11px] leading-relaxed text-ink-500">
        勾选的库才会参与检索；每次提问都会标出答案引用了哪个库。
      </p>

      <Dialog
        open={deleteTarget !== null}
        title={`删除知识库「${deleteTarget?.name ?? ''}」`}
        message={`其中的 ${deleteTarget?.chunk_count ?? 0} 段索引会被一并删除，且无法恢复。`}
        confirmLabel="删除"
        danger
        onConfirm={() => void confirmDelete()}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  )
}
