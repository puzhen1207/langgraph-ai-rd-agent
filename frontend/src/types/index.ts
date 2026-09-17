/**
 * Frontend domain types.
 *
 * Wire shapes come from `api.generated.ts`, which openapi-typescript derives
 * from the backend's OpenAPI schema. Regenerate with:
 *
 *   python backend/scripts/export_openapi.py   # backend -> openapi.json
 *   cd frontend && npm run api:types           # openapi.json -> api.generated.ts
 *
 * Never redeclare a backend field by hand: a server-side rename would compile
 * cleanly here and only surface as `undefined` at runtime.
 */
import type { components } from './api.generated'

type ApiSchemas = components['schemas']

/** Research intents the router actually returns. */
export type IntentType =
  | 'concept_qa'
  | 'literature_review'
  | 'method_compare'
  | 'academic_writing'

/** The backend types `intent` as a plain string, so the union is narrowed here. */
export type ChatRequest = ApiSchemas['ChatRequest']
export type ChatResponse = Omit<ApiSchemas['ChatResponse'], 'intent'> & {
  intent: IntentType
}
export type DocumentStats = ApiSchemas['DocumentStatsResponse']
export type UploadResponse = ApiSchemas['DocumentUploadResponse']
export type HealthResponse = ApiSchemas['HealthResponse']
export type ServiceStatus = ApiSchemas['ServiceStatus']
export type ChunkItem = ApiSchemas['ChunkItem']
/** A named, user-created partition of the vector collection. */
export type KnowledgeBaseInfo = ApiSchemas['KnowledgeBaseInfo']
/** Where one cited passage came from. */
export type SourceRef = ApiSchemas['SourceRef']
/** One row in the sidebar's conversation list. */
export type SessionSummary = ApiSchemas['SessionSummary']

/** UI-only view model for a rendered bubble. Not part of the API contract. */
export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  intent?: IntentType
  timestamp: Date
  retrievedCount?: number
  /** Documents this answer drew on, most relevant first, each tagged with its base. */
  sources?: SourceRef[]
  isStreaming?: boolean
  isValid?: boolean
  validationReason?: string
}

/** One of the four research capabilities, with example questions. */
export interface ResearchMode {
  id: IntentType
  label: string
  description: string
  /** What the answer will be structured like, so the user knows what to expect. */
  answerShape: string
  examples: string[]
}

/** A cross-domain sample question shown on the empty state. */
export interface DomainExample {
  domain: string
  question: string
}
