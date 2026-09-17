import { ChevronRight } from 'lucide-react'
import { RESEARCH_MODES } from '../researchModes'
import type { IntentType } from '../types'

interface Props {
  activeIntent: IntentType | null
  onExampleClick: (example: string) => void
}

/**
 * The four research capabilities.
 *
 * Each mode shows one example rather than three: the sidebar is a map of what
 * the assistant does, and the empty state is where a spread of domains belongs.
 */
export default function ResearchModes({ activeIntent, onExampleClick }: Props) {
  return (
    <div className="px-4 py-4">
      <p className="text-[11px] font-medium tracking-[0.2em] text-ink-600 mb-3">
        研究能力
      </p>

      <div className="space-y-1.5">
        {RESEARCH_MODES.map(mode => {
          const isActive = activeIntent === mode.id
          const Icon = mode.icon
          return (
            <div
              key={mode.id}
              className={`rounded-lg border px-3 py-2.5 transition-colors ${
                isActive
                  ? 'border-brand-200 bg-brand-50'
                  : 'border-transparent bg-transparent hover:bg-paper-200/60'
              }`}
            >
              <div className="flex items-center gap-2.5">
                <Icon
                  size={15}
                  strokeWidth={1.75}
                  className={isActive ? 'text-brand-600' : 'text-ink-600'}
                />
                <span
                  className={`text-[13px] font-medium ${
                    isActive ? 'text-brand-700' : 'text-ink-700'
                  }`}
                >
                  {mode.label}
                </span>
              </div>

              <p className="mt-1 pl-[26px] text-[11px] leading-relaxed text-ink-600">
                {mode.description}
              </p>

              <button
                onClick={() => onExampleClick(mode.examples[0])}
                className="mt-1.5 flex w-full items-center gap-1 pl-[26px]
                           text-left text-[11px] leading-relaxed text-ink-500
                           hover:text-brand-600 transition-colors group"
                title={mode.examples[0]}
              >
                <ChevronRight
                  size={11}
                  className="flex-shrink-0 text-ink-500 group-hover:text-brand-500"
                />
                <span className="truncate">{mode.examples[0]}</span>
              </button>
            </div>
          )
        })}
      </div>
    </div>
  )
}
