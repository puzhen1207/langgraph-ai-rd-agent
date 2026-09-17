import { memo } from 'react'
import ReactMarkdown from 'react-markdown'
// PrismLight + explicit language registration: the full Prism build ships
// every language (~600 kB); the curated set below is what research answers
// actually contain. Unknown fences fall back to plain monospace.
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import bash from 'react-syntax-highlighter/dist/esm/languages/prism/bash'
import javascript from 'react-syntax-highlighter/dist/esm/languages/prism/javascript'
import json from 'react-syntax-highlighter/dist/esm/languages/prism/json'
import markdown from 'react-syntax-highlighter/dist/esm/languages/prism/markdown'
import python from 'react-syntax-highlighter/dist/esm/languages/prism/python'
import sql from 'react-syntax-highlighter/dist/esm/languages/prism/sql'
import typescript from 'react-syntax-highlighter/dist/esm/languages/prism/typescript'
import yaml from 'react-syntax-highlighter/dist/esm/languages/prism/yaml'
import remarkGfm from 'remark-gfm'
import { FileText } from 'lucide-react'
import { intentLabel } from '../researchModes'
import type { Message } from '../types'

for (const [name, lang] of Object.entries({
  bash,
  sh: bash,
  javascript,
  js: javascript,
  json,
  markdown,
  md: markdown,
  python,
  py: python,
  sql,
  typescript,
  ts: typescript,
  yaml,
  yml: yaml,
})) {
  SyntaxHighlighter.registerLanguage(name, lang as Parameters<typeof SyntaxHighlighter.registerLanguage>[1])
}

interface Props {
  message: Message
}

function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user'
  const isStreaming = message.isStreaming === true
  const label = intentLabel(message.intent)
  const sources = message.sources ?? []
  // Which bases this answer actually leaned on, in first-cited order. Shown
  // separately from the file chips: with several bases in scope, "which
  // collection answered this" is the question a researcher asks first.
  const baseNames = [...new Set(sources.map(ref => ref.knowledge_base_name))]

  if (isUser) {
    return (
      <div className="mb-6 flex justify-end">
        <div className="max-w-[78%] rounded-lg rounded-tr-sm bg-brand-700 px-4 py-2.5 text-[13px] leading-relaxed text-paper-50">
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="mb-7 flex gap-3">
      <div className="mt-0.5 flex h-6 w-6 flex-shrink-0 items-center justify-center rounded bg-paper-300 font-serif text-[11px] leading-none text-ink-600">
        研
      </div>

      <div className="min-w-0 flex-1">
        {label && (
          <span className="mb-1.5 inline-block rounded border border-brand-100 bg-brand-50 px-1.5 py-0.5 text-[11px] font-medium text-brand-600">
            {label}
          </span>
        )}

        <div className="rounded-lg border border-paper-300 bg-paper-50 px-4 py-3 text-[13px] leading-relaxed text-ink-800">
          <div className="prose prose-sm max-w-none">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code({ className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className || '')
                  if (!match) {
                    return (
                      <code
                        className="rounded bg-paper-200 px-1 py-0.5 text-[12px] text-mark-700"
                        {...props}
                      >
                        {children}
                      </code>
                    )
                  }
                  // Prism re-highlights the whole block on every render, so
                  // fenced code stays plain until the stream finishes.
                  if (isStreaming) {
                    return (
                      <pre className="overflow-x-auto rounded-lg bg-code-bg p-3 text-[12px] text-code-text">
                        <code>{String(children).replace(/\n$/, '')}</code>
                      </pre>
                    )
                  }
                  return (
                    <SyntaxHighlighter
                      style={vscDarkPlus}
                      language={match[1]}
                      PreTag="div"
                      className="rounded-lg text-[12px]"
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                  )
                },
                table({ children }) {
                  return (
                    <div className="overflow-x-auto">
                      <table className="min-w-full border-collapse text-[12px]">{children}</table>
                    </div>
                  )
                },
              }}
            >
              {message.content}
            </ReactMarkdown>

            {isStreaming && (
              <span className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse bg-brand-500 align-middle" />
            )}
          </div>
        </div>

        {sources.length > 0 ? (
          <div className="mt-2 space-y-1.5">
            <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-ink-600">
              <span>引用 {sources.length} 段 · 来自</span>
              {baseNames.map(name => (
                <span key={name} className="citation-kb">
                  {name}
                </span>
              ))}
            </div>
            <div className="flex flex-wrap gap-1.5">
              {sources.map(ref => (
                <span
                  key={`${ref.knowledge_base_id}/${ref.document}`}
                  className="citation"
                  title={`${ref.document}\n知识库：${ref.knowledge_base_name}`}
                >
                  <FileText size={10} strokeWidth={1.75} />
                  {ref.document}
                </span>
              ))}
            </div>
          </div>
        ) : message.retrievedCount !== undefined && message.retrievedCount > 0 ? (
          // Retrieval ran but every passage was below the relevance threshold
          // (or there genuinely was nothing on the topic). Without a hint the
          // answer would look like the assistant refused to quote anything, so
          // call it out: nothing from the corpus matched strongly enough.
          <div className="mt-2 text-[11px] text-ink-500">
            本次回答未引用知识库内容（基于通用知识，未经知识库佐证）
          </div>
        ) : null}

        <div className="mt-1.5 flex items-center gap-2 text-[11px] text-ink-500">
          <span>
            {message.timestamp.toLocaleTimeString('zh-CN', {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </span>
          {message.isValid === false && (
            <span title={message.validationReason} className="text-mark-600">
              · 未通过质量校验
            </span>
          )}
        </div>
      </div>
    </div>
  )
}

/**
 * Memoised: `useChat` rebuilds the message array on every streamed frame while
 * keeping untouched message objects referentially stable. Without this, each
 * token would re-parse every earlier message as Markdown.
 */
export default memo(MessageBubble)
