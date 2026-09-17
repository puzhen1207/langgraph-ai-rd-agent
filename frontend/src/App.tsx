import { useEffect, useState } from 'react'
import ChatInterface from './components/ChatInterface'
import Sidebar from './components/Sidebar'
import { useSessions } from './hooks/useSessions'
import { useKnowledgeBases } from './hooks/useKnowledgeBases'
import { useTheme } from './hooks/useTheme'
import { healthApi } from './services/api'
import type { IntentType } from './types'

export default function App() {
  const {
    sessions,
    activeSessionId,
    messages,
    isLoading,
    currentNode,
    sendMessage,
    createSession,
    switchSession,
    stopStreaming,
  } = useSessions()
  const kb = useKnowledgeBases()
  const { theme, toggleTheme } = useTheme()
  const [isConnected, setIsConnected] = useState(false)
  // On narrow viewports the sidebar becomes a drawer; this flag is its open
  // state (irrelevant on md+ where the sidebar is always laid out statically).
  const [sidebarOpen, setSidebarOpen] = useState(false)

  // Drives the active-mode highlight in the sidebar and the answer-shape hint
  // above the composer.
  const activeIntent: IntentType | null =
    [...messages].reverse().find(m => m.role === 'assistant' && m.intent)?.intent ?? null

  useEffect(() => {
    healthApi
      .check()
      .then(res => setIsConnected(res.status === 'ok' || res.status === 'degraded'))
      .catch(() => setIsConnected(false))
  }, [])

  // The knowledge-base scope is read at send time rather than captured, so
  // changing the selection immediately affects the next question.
  const handleSend = (query: string) => {
    void sendMessage(query, true, kb.scopeIds)
  }

  return (
    <div className="flex h-[100dvh] overflow-hidden bg-paper-100">
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/50 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        activeIntent={activeIntent}
        activeSessionId={activeSessionId}
        sessions={sessions}
        onSelectSession={id => {
          switchSession(id)
          setSidebarOpen(false)
        }}
        onNewConversation={() => {
          createSession()
          setSidebarOpen(false)
        }}
        onExampleClick={example => {
          handleSend(example)
          setSidebarOpen(false)
        }}
        isConnected={isConnected}
        kb={kb}
        theme={theme}
        onToggleTheme={toggleTheme}
      />
      <main className="min-w-0 flex-1">
        <ChatInterface
          messages={messages}
          isLoading={isLoading}
          currentNode={currentNode}
          onSend={handleSend}
          onStop={stopStreaming}
          onToggleSidebar={() => setSidebarOpen(true)}
          activeIntent={activeIntent}
          scopeLabel={kb.scopeLabel}
          hasDocuments={kb.totalChunks > 0}
        />
      </main>
    </div>
  )
}