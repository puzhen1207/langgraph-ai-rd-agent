import { Library, Lightbulb, PenLine, Scale, type LucideIcon } from 'lucide-react'
import type { DomainExample, IntentType, ResearchMode } from './types'

export interface ResearchModeEntry extends ResearchMode {
  icon: LucideIcon
}

/**
 * The single source for what this assistant can do.
 *
 * Kept in one place because three views need pieces of it: the sidebar renders
 * the modes, the message bubble labels each answer by intent, and the empty
 * state draws on the example questions.
 */
export const RESEARCH_MODES: ResearchModeEntry[] = [
  {
    id: 'concept_qa',
    label: '概念与原理',
    description: '术语 · 机制 · 成因',
    answerShape: '先给结论，再解释“为什么”与“如何起作用”，标注依据来源',
    icon: Lightbulb,
    examples: [
      'Transformer 的自注意力机制解决了什么问题？',
      'CRISPR-Cas9 的脱靶效应如何评估？',
      '货币政策通过哪些渠道传导到实体经济？',
    ],
  },
  {
    id: 'literature_review',
    label: '文献综述',
    description: '研究现状 · 路线 · 争议',
    answerShape: '按“脉络 / 路线 / 共识与分歧 / 未决问题 / 覆盖缺口”组织',
    icon: Library,
    examples: [
      'Transformer 的研究现状和主要技术路线有哪些？',
      '因果推断在经济学中的研究进展如何？',
      '单细胞测序目前有哪些主要技术方向？',
    ],
  },
  {
    id: 'method_compare',
    label: '方法对比',
    description: '选型 · 取舍 · 边界',
    answerShape: '先给对比表格，再逐项给依据，最后给有条件的选型建议',
    icon: Scale,
    examples: [
      '对比注意力机制与循环结构的区别和适用场景',
      '随机对照试验与断点回归各自适合什么情况？',
      '比较 CRISPR 与 TALEN 的优缺点',
    ],
  },
  {
    id: 'academic_writing',
    label: '学术写作',
    description: '润色 · 翻译 · 摘要 · 审稿回复',
    answerShape: '先给修改稿，再列出主要修改点与理由，保留原意与限定条件',
    icon: PenLine,
    examples: [
      '帮我润色这段摘要，使其更符合期刊语体',
      '把这段中文学术表述翻译成英文',
      '根据摘要拟几个更准确的论文标题',
    ],
  },
]

const LABELS = new Map(RESEARCH_MODES.map(mode => [mode.id, mode.label]))

/** Human-readable label for an intent id, falling back to the raw id. */
export function intentLabel(intent: IntentType | undefined): string | null {
  if (!intent) return null
  return LABELS.get(intent) ?? intent
}

/**
 * Cross-domain prompts for the empty state.
 *
 * Deliberately spans several disciplines: the point of the product is that the
 * corpus, not the assistant, determines the field.
 */
export const DOMAIN_EXAMPLES: DomainExample[] = [
  { domain: '深度学习', question: 'Transformer 的自注意力机制解决了什么问题？' },
  { domain: '生物医学', question: 'CRISPR-Cas9 的脱靶效应如何评估？' },
  { domain: '经济学', question: '货币政策通过哪些渠道传导到实体经济？' },
  { domain: '研究方法', question: '随机对照试验与断点回归各自适合什么情况？' },
  { domain: '软件工程', question: 'Android 的长列表为什么要做懒加载？' },
  { domain: '数据库', question: 'MVCC 和事务隔离级别是什么关系？' },
]
