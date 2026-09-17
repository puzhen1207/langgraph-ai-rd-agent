import { useEffect, useRef, useState } from 'react'
import {
  ChevronDown,
  ChevronRight,
  Eye,
  EyeOff,
  FileText,
  KeyRound,
  Loader2,
  Trash2,
  Upload,
} from 'lucide-react'
import Dialog from './Dialog'
import { describeError, documentsApi, setAdminKey } from '../services/api'
import type { ChunkItem, KnowledgeBaseInfo } from '../types'

interface Props {
  knowledgeBases: KnowledgeBaseInfo[]
  uploadTargetId: string
  onUploadTargetChange: (id: string) => void
  /** Called after the contents change so the base list can refresh its counts. */
  onChanged: () => void
}

export default function DocumentUpload({
  knowledgeBases,
  uploadTargetId,
  onUploadTargetChange,
  onChanged,
}: Props) {
  const [uploading, setUploading] = useState(false)
  const [lastResult, setLastResult] = useState('')
  const [chunks, setChunks] = useState<ChunkItem[]>([])
  const [showChunks, setShowChunks] = useState(false)
  const [expandedChunk, setExpandedChunk] = useState<string | null>(null)
  const [confirmClear, setConfirmClear] = useState(false)
  const [keyDialogOpen, setKeyDialogOpen] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const target = knowledgeBases.find(kb => kb.id === uploadTargetId) ?? null

  // A different target means different contents; keeping the old list would
  // show passages that are no longer reachable from the current selection.
  useEffect(() => {
    setChunks([])
    setExpandedChunk(null)
  }, [uploadTargetId])

  const fetchChunks = async (kbId: string) => {
    try {
      setChunks(await documentsApi.getChunks(kbId))
    } catch {
      setChunks([])
    }
  }

  const handleToggleChunks = async () => {
    if (!showChunks && chunks.length === 0 && target) await fetchChunks(target.id)
    setShowChunks(v => !v)
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (!target) {
      setLastResult('请先新建一个知识库，再上传文档')
      return
    }
    setUploading(true)
    setLastResult('')
    try {
      const res = await documentsApi.upload(file, target.id)
      setLastResult(
        `${res.filename} → 「${res.knowledge_base_name}」：新增 ${res.chunks_added} 段，共 ${res.total_docs} 段`
      )
      onChanged()
      if (showChunks) await fetchChunks(target.id)
    } catch (err) {
      setLastResult(`上传失败：${describeError(err)}`)
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handleClearCurrent = () => setConfirmClear(true)

  const confirmClearTarget = async () => {
    if (!target) return
    setConfirmClear(false)
    try {
      const res = await documentsApi.clearKnowledgeBase(target.id)
      setChunks([])
      setShowChunks(false)
      setLastResult(`${target.name}：${res.message}`)
      onChanged()
    } catch (err) {
      setLastResult(`清空失败：${describeError(err)}`)
    }
  }

  const submitAdminKey = (key: string) => {
    setKeyDialogOpen(false)
    setAdminKey(key)
    setLastResult(key.trim() ? '管理员密钥已保存到当前会话' : '管理员密钥已清除')
  }

  const grouped = chunks.reduce<Record<string, ChunkItem[]>>((acc, chunk) => {
    ;(acc[chunk.source] ??= []).push(chunk)
    return acc
  }, {})

  return (
    <div className="border-t border-paper-300 px-4 py-4">
      <div className="mb-2 flex items-center justify-between">
        <p className="text-[11px] font-medium tracking-[0.2em] text-ink-600">
          添加文献
        </p>
        {target && target.chunk_count > 0 && (
          <button
            onClick={handleToggleChunks}
            className="flex items-center gap-1 text-[11px] text-ink-600 transition-colors hover:text-brand-600"
          >
            {showChunks ? <EyeOff size={11} /> : <Eye size={11} />}
            {showChunks ? '收起' : `查看 ${target.chunk_count} 段`}
          </button>
        )}
      </div>

      {showChunks && (
        <div className="mb-3 max-h-56 overflow-y-auto rounded-lg border border-paper-300 bg-paper-50 text-[11px]">
          {Object.entries(grouped).map(([source, items]) => (
            <div key={source}>
              <div className="sticky top-0 flex items-center gap-1.5 border-b border-paper-200 bg-paper-200/80 px-3 py-1.5 text-ink-600 backdrop-blur">
                <FileText size={11} strokeWidth={1.75} className="flex-shrink-0 text-mark-500" />
                <span className="truncate font-medium">{source}</span>
                <span className="ml-auto flex-shrink-0 text-ink-600">{items.length} 段</span>
              </div>
              {items.map((chunk, idx) => (
                <div key={chunk.id} className="border-b border-paper-100 last:border-0">
                  <button
                    className="flex w-full items-center gap-2 px-3 py-1.5 text-left transition-colors hover:bg-paper-200/60"
                    onClick={() => setExpandedChunk(expandedChunk === chunk.id ? null : chunk.id)}
                  >
                    {expandedChunk === chunk.id ? (
                      <ChevronDown size={11} className="flex-shrink-0 text-ink-500" />
                    ) : (
                      <ChevronRight size={11} className="flex-shrink-0 text-ink-500" />
                    )}
                    <span className="flex-1 truncate text-ink-500">
                      {chunk.content.slice(0, 52).replace(/\n/g, ' ')}…
                    </span>
                    <span className="flex-shrink-0 text-ink-500">#{idx + 1}</span>
                  </button>
                  {expandedChunk === chunk.id && (
                    <div className="whitespace-pre-wrap bg-paper-100 px-3 pb-2.5 pl-6 leading-relaxed text-ink-500">
                      {chunk.content}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>
      )}

      {knowledgeBases.length === 0 ? (
        <p className="rounded-md bg-paper-200/70 px-2 py-2 text-[11px] leading-relaxed text-ink-500">
          先在上方新建一个知识库，再上传文档。每个知识库是一个独立领域，
          提问时可以选择检索哪些库。
        </p>
      ) : (
        <>
          <label className="mb-1.5 block text-[11px] text-ink-600">上传到</label>
          <select
            value={uploadTargetId}
            onChange={e => onUploadTargetChange(e.target.value)}
            className="w-full rounded-md border border-paper-300 bg-paper-50 px-2 py-1.5 text-[11px] text-ink-700 outline-none transition-colors focus:border-brand-500"
          >
            {knowledgeBases.map(kb => (
              <option key={kb.id} value={kb.id}>
                {kb.name}（{kb.document_count} 篇）
              </option>
            ))}
          </select>

          <label
            className={`
              mt-2 flex w-full cursor-pointer items-center justify-center gap-2 rounded-lg border border-dashed
              px-3 py-2.5 text-[11px] font-medium transition-colors
              ${uploading
                ? 'cursor-not-allowed border-paper-300 text-ink-500'
                : 'border-brand-200 text-brand-600 hover:bg-brand-50'
              }
            `}
          >
            {uploading ? (
              <>
                <Loader2 size={12} strokeWidth={1.75} className="animate-spin" /> 索引中…
              </>
            ) : (
              <>
                <Upload size={12} strokeWidth={1.75} /> 选择文件
              </>
            )}
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept=".md,.txt,.kt,.java,.html,.wiki,.docx,.pdf"
              onChange={handleFileChange}
              disabled={uploading}
            />
          </label>

          <p className="mt-1.5 text-center text-[10px] text-ink-500">
            支持 .md .txt .pdf .docx .html .wiki .kt .java
          </p>
        </>
      )}

      <button
        onClick={() => setKeyDialogOpen(true)}
        className="mt-1.5 flex w-full items-center justify-center gap-1 py-1 text-[11px] text-ink-600 transition-colors hover:text-brand-600"
      >
        <KeyRound size={11} /> 配置管理员密钥
      </button>

      {lastResult && (
        <p className="mt-2 rounded-md bg-paper-200/70 px-2 py-1.5 text-[11px] leading-relaxed text-ink-600">
          {lastResult}
        </p>
      )}

      {target && target.chunk_count > 0 && (
        <button
          onClick={handleClearCurrent}
          className="mt-2 flex w-full items-center justify-center gap-1 py-1 text-[11px] text-ink-600 transition-colors hover:text-red-500"
        >
          <Trash2 size={11} /> 清空「{target.name}」的索引
        </button>
      )}

      <Dialog
        open={confirmClear}
        title={`清空知识库「${target?.name ?? ''}」`}
        message={`其中 ${target?.chunk_count ?? 0} 段索引会被删除，知识库本身保留。`}
        confirmLabel="清空"
        danger
        onConfirm={() => void confirmClearTarget()}
        onCancel={() => setConfirmClear(false)}
      />

      <Dialog
        open={keyDialogOpen}
        title="配置管理员密钥"
        message="仅保存在当前浏览器会话中；留空则清除。"
        confirmLabel="保存"
        input={{
          type: 'password',
          placeholder: '管理员密钥',
          initialValue: '',
        }}
        onConfirm={submitAdminKey}
        onCancel={() => setKeyDialogOpen(false)}
      />
    </div>
  )
}
