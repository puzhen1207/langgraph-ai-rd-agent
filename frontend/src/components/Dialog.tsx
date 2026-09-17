import { useEffect, useRef, useState } from 'react'

interface DialogInput {
  /** 'password' hides the typed value; use for secrets. */
  type?: 'text' | 'password'
  placeholder?: string
  /** Pre-fills the input when the dialog opens. */
  initialValue?: string
}

interface Props {
  open: boolean
  title: string
  /** Supporting copy shown under the title; newlines are preserved. */
  message?: string
  confirmLabel: string
  /** Red confirm button for irreversible actions. */
  danger?: boolean
  /**
   * When present the dialog shows a text input and passes its value to
   * onConfirm (empty string allowed). Without it onConfirm receives ''.
   */
  input?: DialogInput
  onConfirm: (value: string) => void
  onCancel: () => void
}

/**
 * Inline replacement for window.confirm / window.prompt.
 *
 * Native dialogs clash with the paper-and-ink visual language and prompt()
 * cannot mask its input, so a secret typed into it is displayed in plain
 * text. This component keeps everything inside the app's own design tokens
 * and supports a password field. Enter submits, Escape cancels.
 */
export default function Dialog({
  open,
  title,
  message,
  confirmLabel,
  danger,
  input,
  onConfirm,
  onCancel,
}: Props) {
  const [value, setValue] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) setValue(input?.initialValue ?? '')
  }, [open, input?.initialValue])

  useEffect(() => {
    if (open) inputRef.current?.focus()
  }, [open])

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      role="dialog"
      aria-modal="true"
      aria-label={title}
    >
      <div className="w-full max-w-sm rounded-lg border border-paper-300 bg-paper-50 p-5 shadow-xl">
        <h3 className="font-serif text-[14px] leading-snug text-ink-900">{title}</h3>
        {message && (
          <p className="mt-2 whitespace-pre-line text-[12px] leading-relaxed text-ink-500">
            {message}
          </p>
        )}
        {input && (
          <input
            ref={inputRef}
            type={input.type ?? 'text'}
            value={value}
            placeholder={input.placeholder}
            onChange={e => setValue(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter') onConfirm(value)
              if (e.key === 'Escape') onCancel()
            }}
            className="mt-3 w-full rounded-md border border-paper-300 bg-paper-100 px-2.5 py-1.5
                       text-[12px] text-ink-800 outline-none transition-colors
                       placeholder:text-ink-400 focus:border-brand-500"
          />
        )}
        <div className="mt-4 flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-md border border-paper-300 px-3 py-1.5 text-[12px] text-ink-600
                       transition-colors hover:bg-paper-100"
          >
            取消
          </button>
          <button
            type="button"
            onClick={() => onConfirm(value)}
            autoFocus={!input}
            className={`rounded-md px-3 py-1.5 text-[12px] text-paper-50 transition-colors ${
              danger
                ? 'bg-red-600 hover:bg-red-700'
                : 'bg-brand-700 hover:bg-brand-800'
            }`}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
