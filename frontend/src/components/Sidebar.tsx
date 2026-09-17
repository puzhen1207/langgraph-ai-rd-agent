import { useMemo } from 'react'
import { MessageSquare, Moon, PanelLeftClose, Plus, Sun } from 'lucide-react'
import DocumentUpload from './DocumentUpload'
import KnowledgeBasePanel from './KnowledgeBasePanel'
import ResearchModes from './ResearchModes'
import type { KnowledgeBasesController } from '../hooks/useKnowledgeBases'
import type { IntentType, SessionSummary } from '../types'
import type { Theme } from '../hooks/useTheme'

interface Props {
  /** Drawer open state; only meaningful below md where the sidebar overlays. */
  open: boolean
  onClose: () => void
  activeIntent: IntentType | null
  activeSessionId: string
  sessions: SessionSummary[]
  onSelectSession: (sessionId: string) => void
  onNewConversation: () => void
  onExampleClick: (example: string) => void
  isConnected: boolean
  kb: KnowledgeBasesController
  theme: Theme
  onToggleTheme: () => void
}

function formatTimestamp(ts: number): string {
  // 0 (or negative) means "no real timestamp" — epoch would render 1970/1/1.
  if (!ts || ts <= 0) return '—'
  const date = new Date(ts * 1000)
  const now = new Date()
  const sameDay =
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate()
  if (sameDay) {
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  }
  const sameYear = date.getFullYear() === now.getFullYear()
  return date.toLocaleDateString('zh-CN', {
    month: 'numeric',
    day: 'numeric',
    ...(sameYear ? {} : { year: 'numeric' }),
  })
}

export default function Sidebar({
  open,
  onClose,
  activeIntent,
  activeSessionId,
  sessions,
  onSelectSession,
  onNewConversation,
  onExampleClick,
  isConnected,
  kb,
  theme,
  onToggleTheme,
}: Props) {
  const sortedSessions = useMemo(() => sessions, [sessions])

  return (
    <aside
      className={`fixed inset-y-0 left-0 z-40 flex w-80 flex-col overflow-hidden
                  border-r border-paper-300 bg-paper-50 transition-transform duration-200 ease-in-out
                  md:static md:z-auto md:translate-x-0
                  ${open ? 'translate-x-0' : '-translate-x-full'}`}
    >
      <div className="border-b border-paper-300 px-5 py-5">
        <div className="flex items-start justify-between">
          <div className="flex items-start gap-2.5">
            <div className="mt-0.5 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded bg-brand-700 font-serif text-[15px] leading-none text-paper-50">
              研
            </div>
            <div className="min-w-0">
              <h1 className="font-serif text-[17px] leading-tight text-ink-900">
                AI 科研助手
              </h1>
              <p className="mt-0.5 text-[11px] text-ink-600">
                多知识库 · 跨领域研究问答
              </p>
            </div>
          </div>
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={onToggleTheme}
              title={theme === 'dark' ? '切换到浅色模式' : '切换到深色模式'}
              aria-label={theme === 'dark' ? '切换到浅色模式' : '切换到深色模式'}
              className="flex h-7 w-7 items-center justify-center rounded-md text-ink-600 transition-colors hover:bg-paper-200 hover:text-ink-900"
            >
              {theme === 'dark' ? <Sun size={15} strokeWidth={1.75} /> : <Moon size={15} strokeWidth={1.75} />}
            </button>
            <button
              type="button"
              onClick={onClose}
              title="收起侧边栏"
              aria-label="收起侧边栏"
              className="flex h-7 w-7 items-center justify-center rounded-md text-ink-600 transition-colors hover:bg-paper-200 hover:text-ink-900 md:hidden"
            >
              <PanelLeftClose size={15} strokeWidth={1.75} />
            </button>
          </div>
        </div>

        <div className="mt-3 flex items-center gap-1.5">
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              isConnected ? 'bg-emerald-500' : 'bg-red-400'
            }`}
          />
          <span className="text-[11px] text-ink-600">
            {isConnected ? '服务在线' : '服务离线'}
          </span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Conversation history sits above research modes so the list is the
            first thing the user reaches for when coming back to the app. */}
        <div className="px-5 pt-5">
          <div className="mb-2 flex items-center justify-between">
            <h2 className="font-serif text-[12px] tracking-[0.2em] text-ink-600">
              对话历史
            </h2>
            <button
              type="button"
              onClick={onNewConversation}
              title="新对话"
              aria-label="新对话"
              className="inline-flex h-6 w-6 items-center justify-center rounded-md border border-paper-300 bg-paper-50 text-ink-600 transition-colors hover:border-brand-500 hover:text-brand-700"
            >
              <Plus className="h-3.5 w-3.5" />
            </button>
          </div>

          {sortedSessions.length === 0 ? (
            <p className="px-1 text-[12px] leading-relaxed text-ink-600">
              还没有对话记录。点上方 <Plus className="inline h-3 w-3 align-text-bottom" /> 开始第一个。
            </p>
          ) : (
            <ul className="space-y-1">
              {sortedSessions.map(s => {
                const active = s.session_id === activeSessionId
                return (
                  <li key={s.session_id}>
                    <button
                      type="button"
                      onClick={() => onSelectSession(s.session_id)}
                      title={s.title}
                      className={`group flex w-full items-start gap-2 rounded-md px-2.5 py-2 text-left transition-colors ${
                        active
                          ? 'bg-brand-100 text-ink-900'
                          : 'text-ink-700 hover:bg-paper-100'
                      }`}
                    >
                      <MessageSquare
                        className={`mt-0.5 h-3.5 w-3.5 flex-shrink-0 ${
                          active ? 'text-brand-700' : 'text-ink-600'
                        }`}
                      />
                      <div className="min-w-0 flex-1">
                        <p
                          className={`truncate text-[13px] leading-snug ${
                            active ? 'font-medium' : ''
                          }`}
                        >
                          {s.title || '新对话'}
                        </p>
                        <p className="mt-0.5 flex items-center gap-1.5 text-[10.5px] text-ink-600">
                          <span>{formatTimestamp(s.updated_at)}</span>
                          {s.message_count > 0 && (
                            <>
                              <span>·</span>
                              <span>{s.message_count} 条</span>
                            </>
                          )}
                        </p>
                      </div>
                    </button>
                  </li>
                )
              })}
            </ul>
          )}
        </div>

        <ResearchModes activeIntent={activeIntent} onExampleClick={onExampleClick} />
        <KnowledgeBasePanel
          items={kb.items}
          selectedIds={kb.selectedIds}
          loading={kb.loading}
          onToggle={kb.toggle}
          onSelectAll={kb.selectAll}
          onCreate={kb.create}
          onRename={kb.rename}
          onDelete={kb.remove}
        />
      </div>

      <DocumentUpload
        knowledgeBases={kb.items}
        uploadTargetId={kb.uploadTargetId}
        onUploadTargetChange={kb.setUploadTargetId}
        onChanged={kb.refresh}
      />
    </aside>
  )
}