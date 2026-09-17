import { useEffect, useRef, useState } from 'react'
import { ArrowUp, Minus, PanelLeft } from 'lucide-react'
import MessageBubble from './MessageBubble'
import { DOMAIN_EXAMPLES, RESEARCH_MODES } from '../researchModes'
import type { IntentType, Message } from '../types'

const NODE_LABELS: Record<string, string> = {
  query_analyze: '理解问题',
  intent_router: '判定研究意图',
  rag_retrieve: '检索文献',
  rerank: '精排片段',
  memory_inject: '载入上下文',
  llm_generate: '撰写回答',
  response_check: '校验依据',
}

interface Props {
  messages: Message[]
  isLoading: boolean
  currentNode: string
  onSend: (query: string) => void
  onStop: () => void
  /** Opens the sidebar drawer; the button only renders below md. */
  onToggleSidebar: () => void
  activeIntent: IntentType | null
  /** Human-readable summary of the knowledge bases currently in scope. */
  scopeLabel: string
  /** False before anything has been indexed, which changes the empty state. */
  hasDocuments: boolean
}

export default function ChatInterface({
  messages,
  isLoading,
  currentNode,
  onSend,
  onStop,
  onToggleSidebar,
  activeIntent,
  scopeLabel,
  hasDocuments,
}: Props) {
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    // Restarting a smooth scroll on every streamed frame fights the browser and
    // makes the view stutter, so jump directly while tokens are arriving.
    const lastMessage = messages[messages.length - 1]
    bottomRef.current?.scrollIntoView({
      behavior: lastMessage?.isStreaming ? 'auto' : 'smooth',
    })
  }, [messages])

  const handleSend = () => {
    const q = input.trim()
    if (!q || isLoading) return
    onSend(q)
    setInput('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px'
  }

  const isEmpty = messages.length === 0
  const activeMode = RESEARCH_MODES.find(mode => mode.id === activeIntent) ?? null

  return (
    <div className="relative flex h-full flex-col bg-paper-100">
      <button
        type="button"
        onClick={onToggleSidebar}
        title="打开侧边栏"
        aria-label="打开侧边栏"
        className="absolute left-3 top-3 z-10 flex h-8 w-8 items-center justify-center rounded-md
                   border border-paper-300 bg-paper-50 text-ink-600 shadow-sm transition-colors
                   hover:border-brand-400 hover:text-brand-700 md:hidden"
      >
        <PanelLeft size={15} strokeWidth={1.75} />
      </button>

      <div className="flex-1 overflow-y-auto px-4 pb-8 pt-14 md:px-8 md:pt-8">
        {isEmpty ? (
          <div className="mx-auto flex h-full max-w-2xl flex-col justify-center">
            <h2 className="font-serif text-2xl leading-snug text-ink-900">
              跨领域科研问答
            </h2>
            <p className="mt-2 text-[13px] leading-relaxed text-ink-500">
              上传你的文献与资料，助手只基于它们回答，并标注依据来自哪个知识库。
              不限定学科——领域由你建立的库决定。
            </p>

            {!hasDocuments ? (
              <div className="mt-7 rounded-lg border border-dashed border-brand-200 bg-brand-50/50 px-5 py-5">
                <p className="text-[13px] font-medium text-brand-700">
                  第一步：建一个知识库，再上传文献
                </p>
                <ol className="mt-2 space-y-1 text-[12px] leading-relaxed text-ink-600">
                  <li>1. 在左侧「知识库」里点「新建知识库」，给它起个领域名，如「深度学习文献」。</li>
                  <li>2. 在「添加文献」里选好目标库，上传 PDF / Markdown / 文本。</li>
                  <li>3. 提问后，答案下方会列出引用了哪些文件、来自哪个库。</li>
                </ol>
              </div>
            ) : (
              <div className="mt-7 grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                {DOMAIN_EXAMPLES.map(example => (
                  <button
                    key={example.question}
                    onClick={() => onSend(example.question)}
                    className="group rounded-lg border border-paper-300 bg-paper-50 px-3.5 py-3
                               text-left transition-colors hover:border-brand-200 hover:bg-brand-50"
                  >
                    <span className="text-[10px] font-medium tracking-[0.2em] text-ink-500 group-hover:text-brand-500">
                      {example.domain}
                    </span>
                    <p className="mt-1 text-[12px] leading-relaxed text-ink-700">
                      {example.question}
                    </p>
                  </button>
                ))}
              </div>
            )}

            <p className="mt-6 text-[11px] leading-relaxed text-ink-500">
              {hasDocuments
                ? '示例问题基于内置示例语料；换成你自己的知识库后，直接提问你的领域即可。'
                : '已内置示例语料，也可以直接提问试试，例如「Transformer 的自注意力机制解决了什么问题？」'}
            </p>
          </div>
        ) : (
          <div className="mx-auto max-w-3xl">
            {messages.map(msg => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {isLoading && currentNode && (
        <div className="px-4 pb-2 md:px-8">
          <div className="mx-auto flex max-w-3xl items-center gap-2.5 rounded-md border border-paper-300 bg-paper-50 px-3 py-2">
            <span className="text-[11px] text-brand-600">
              {NODE_LABELS[currentNode] ?? currentNode}
            </span>
            <div className="ml-auto flex gap-1">
              {Object.keys(NODE_LABELS).map(node => (
                <span
                  key={node}
                  className={`h-0.5 w-5 rounded-full transition-colors ${
                    node === currentNode ? 'bg-brand-500' : 'bg-paper-300'
                  }`}
                />
              ))}
            </div>
          </div>
        </div>
      )}

      <div className="border-t border-paper-300 bg-paper-50 px-4 py-4 md:px-8">
        <div className="mx-auto max-w-3xl">
          <div className="flex items-end gap-2.5 rounded-lg border border-paper-300 bg-paper-100 px-3.5 py-2.5 transition-colors focus-within:border-brand-500">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={handleTextareaChange}
              onKeyDown={handleKeyDown}
              placeholder="输入研究问题…（Enter 发送，Shift+Enter 换行）"
              rows={1}
              className="max-h-40 flex-1 resize-none bg-transparent text-[13px] text-ink-800 outline-none placeholder:text-ink-400"
              disabled={isLoading}
            />
            {isLoading ? (
              <button
                onClick={onStop}
                title="停止生成"
                className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-md border border-paper-300 bg-paper-50 text-ink-500 transition-colors hover:text-red-500"
              >
                <Minus size={15} strokeWidth={1.75} />
              </button>
            ) : (
              <button
                onClick={handleSend}
                disabled={!input.trim()}
                title="发送"
                className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-md bg-brand-700 text-paper-50 transition-colors hover:bg-brand-800 disabled:cursor-not-allowed disabled:bg-paper-300 disabled:text-ink-400"
              >
                <ArrowUp size={15} strokeWidth={1.75} />
              </button>
            )}
          </div>

          <p className="mt-2 text-center text-[11px] leading-relaxed text-ink-500">
            检索范围：<span className="text-ink-500">{scopeLabel}</span>
          </p>
          {activeMode && (
            <p className="mt-0.5 text-center text-[11px] leading-relaxed text-ink-500">
              <span className="text-ink-500">{activeMode.label}</span>
              {'：'}
              {activeMode.answerShape}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
