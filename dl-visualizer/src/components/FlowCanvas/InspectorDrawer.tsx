import type { CSSProperties } from 'react'
import type { BenchmarkEntry, CodexStatus, SamplePreview } from '../../lib/api'
import type { GraphDiagnostic } from '../../lib/graphDiagnostics'

interface Props {
  activeTab: 'sample' | 'diagnostics'
  onTabChange: (tab: 'sample' | 'diagnostics') => void
  samplePreview?: SamplePreview | null
  benchmark?: BenchmarkEntry | null
  diagnostics: GraphDiagnostic[]
  graphHealth: {
    level: 'healthy' | 'warning' | 'error'
    label: string
    message: string
    errorCount: number
    warningCount: number
    infoCount: number
  }
  hideExpectedTerminals: boolean
  onToggleHideExpectedTerminals: () => void
  onAutoLayout: () => void
  onNormalizeInputShapes: () => void
  onClearGraph: () => void
  codexStatus?: CodexStatus | null
  codexValidation: { status: 'idle' | 'running' | 'done' | 'error'; message: string }
  onValidateCodex: () => void
  codexInstruction: string
  onCodexInstructionChange: (value: string) => void
  codexEdit: { status: 'idle' | 'running' | 'done' | 'error'; message: string; diffSummary?: string }
  onApplyCodexEdit: () => void
  traceMode?: string
  exactness?: string
  unsupportedReason?: string
  constructorStrategy?: string
  constructorCallable?: string
  inputStrategy?: string
  runtimeMs?: number | null
}

export default function InspectorDrawer({
  // activeTab, onTabChange, samplePreview, benchmark unused (moved to floating overlay)
  diagnostics,
  graphHealth,
  hideExpectedTerminals,
  onToggleHideExpectedTerminals,
  onAutoLayout,
  onNormalizeInputShapes,
  onClearGraph,
  codexStatus,
  codexValidation,
  onValidateCodex,
  codexInstruction,
  onCodexInstructionChange,
  codexEdit,
  onApplyCodexEdit,
}: Props) {
  const visibleDiagnostics = hideExpectedTerminals
    ? diagnostics.filter((item) => !item.expectedTerminal)
    : diagnostics

  return (
    <div style={{
      borderTop: '1px solid #1e2535',
      background: '#10141f',
      minHeight: 230,
      maxHeight: 230,
      display: 'flex',
      flexDirection: 'column',
      flexShrink: 0,
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '8px 16px',
        borderBottom: '1px solid #1e2535',
        background: '#131927',
      }}>
        <span
          data-testid="drawer-tab-diagnostics"
          style={{ fontSize: 11, fontWeight: 600, color: '#94a3b8', padding: '3px 0' }}
        >
          Diagnostics ({visibleDiagnostics.length})
        </span>
        <div style={{ flex: 1 }} />
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#94a3b8' }}>
          <input
            type="checkbox"
            checked={hideExpectedTerminals}
            onChange={onToggleHideExpectedTerminals}
          />
          Hide expected terminals
        </label>
      </div>

      <div style={{ flex: 1, overflow: 'auto', padding: 16 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: 16 }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {visibleDiagnostics.length === 0 ? (
                <div style={{ fontSize: 12, color: '#34d399' }}>No visible diagnostics for the current graph.</div>
              ) : (
                visibleDiagnostics.map((diagnostic) => (
                  <div
                    key={`${diagnostic.nodeId}-${diagnostic.code}`}
                    data-testid="diagnostic-item"
                    data-severity={diagnostic.severity}
                    data-code={diagnostic.code}
                    data-node-label={diagnostic.nodeLabel}
                    data-title={diagnostic.title}
                    data-detail={diagnostic.detail}
                    style={{
                      border: `1px solid ${severityColor(diagnostic.severity)}55`,
                      background: '#0b1220',
                      borderRadius: 10,
                      padding: '10px 12px',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{
                        fontSize: 10,
                        fontWeight: 700,
                        color: severityColor(diagnostic.severity),
                        textTransform: 'uppercase',
                      }}>
                        {diagnostic.severity}
                      </span>
                      <span style={{ fontSize: 12, color: '#e2e8f0', fontWeight: 600 }}>
                        {diagnostic.nodeLabel}
                      </span>
                    </div>
                    <div style={{ fontSize: 12, color: '#e2e8f0', marginTop: 6 }}>{diagnostic.title}</div>
                    <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4, lineHeight: 1.5 }}>{diagnostic.detail}</div>
                    {diagnostic.recoveryHint ? (
                      <div style={{ fontSize: 11, color: '#cbd5e1', marginTop: 6, lineHeight: 1.5 }}>
                        Recovery: {diagnostic.recoveryHint}
                      </div>
                    ) : null}
                  </div>
                ))
              )}
            </div>
            <div style={{
              border: '1px solid #243146',
              background: '#0b1220',
              borderRadius: 10,
              padding: 12,
              display: 'flex',
              flexDirection: 'column',
              gap: 10,
              height: 'fit-content',
            }}>
              <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                Recovery Actions
              </div>
              <StatusPanel
                label={graphHealth.label}
                level={graphHealth.level}
                message={graphHealth.message}
                meta="Use these actions to recover from common graph issues without losing the current canvas."
              />
              <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 8 }}>
                <button
                  data-testid="recovery-auto-layout"
                  onClick={onAutoLayout}
                  style={actionButtonStyle('#1d4ed8')}
                >
                  Auto layout graph
                </button>
                <button
                  data-testid="recovery-normalize-inputs"
                  onClick={onNormalizeInputShapes}
                  style={actionButtonStyle('#0f766e')}
                >
                  Normalize input shapes
                </button>
                <button
                  data-testid="recovery-clear-graph"
                  onClick={onClearGraph}
                  style={actionButtonStyle('#7c2d12')}
                >
                  Clear canvas
                </button>
              </div>
              <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5 }}>
                Unsupported runtime states are shown with a friendly label above. If runtime exact tracing is unavailable, the UI stays in exploratory static mode and records the reason in Sample Data.
              </div>
              <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.08em', marginTop: 8 }}>
                Codex Exec Path
              </div>
              <InfoRow label="Transport" value={codexStatus?.transport ?? 'checking'} />
              <InfoRow label="Auth" value={codexStatus?.authReady ? `ready (${codexStatus.authMode ?? 'unknown'})` : 'not ready'} />
              <InfoRow label="Binary" value={codexStatus?.binaryPath ?? 'not found'} />
              <button
                data-testid="validate-codex-button"
                onClick={onValidateCodex}
                disabled={codexValidation.status === 'running'}
                style={{
                  padding: '8px 10px',
                  borderRadius: 8,
                  border: 'none',
                  background: codexValidation.status === 'running' ? '#334155' : '#2563eb',
                  color: '#fff',
                  fontSize: 12,
                  cursor: codexValidation.status === 'running' ? 'default' : 'pointer',
                }}
              >
                {codexValidation.status === 'running' ? 'Validating…' : 'Validate official codex exec'}
              </button>
              <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5 }}>
                {codexValidation.message || 'No validation run yet.'}
              </div>
              <div style={{
                marginTop: 6,
                paddingTop: 10,
                borderTop: '1px solid #1e2535',
                display: 'flex',
                flexDirection: 'column',
                gap: 8,
              }}>
                <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                  Source-aware edit
                </div>
                <textarea
                  data-testid="codex-edit-instruction"
                  value={codexInstruction}
                  onChange={(event) => onCodexInstructionChange(event.target.value)}
                  placeholder="Describe a focused edit to the original source file."
                  style={{
                    minHeight: 88,
                    resize: 'vertical',
                    background: '#0f172a',
                    border: '1px solid #334155',
                    color: '#e2e8f0',
                    borderRadius: 8,
                    padding: '8px 10px',
                    fontSize: 12,
                    lineHeight: 1.5,
                  }}
                />
                <button
                  data-testid="codex-apply-button"
                  onClick={onApplyCodexEdit}
                  disabled={codexEdit.status === 'running' || !codexInstruction.trim()}
                  style={{
                    padding: '8px 10px',
                    borderRadius: 8,
                    border: 'none',
                    background: codexEdit.status === 'running' || !codexInstruction.trim() ? '#334155' : '#0f766e',
                    color: '#fff',
                    fontSize: 12,
                    cursor: codexEdit.status === 'running' || !codexInstruction.trim() ? 'default' : 'pointer',
                  }}
                >
                  {codexEdit.status === 'running' ? 'Running Codex edit…' : 'Run source-aware edit in worktree'}
                </button>
                <div data-testid="codex-edit-message" style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5 }}>
                  {codexEdit.message || 'No source-aware edit run yet.'}
                </div>
                {codexEdit.diffSummary ? (
                  <pre
                    data-testid="codex-edit-diff"
                    style={{
                      margin: 0,
                      padding: 10,
                      borderRadius: 8,
                      background: '#020617',
                      border: '1px solid #1e293b',
                      color: '#cbd5e1',
                      fontSize: 10,
                      lineHeight: 1.45,
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      maxHeight: 180,
                      overflow: 'auto',
                    }}
                  >
                    {codexEdit.diffSummary}
                  </pre>
                ) : null}
              </div>
            </div>
          </div>
      </div>
    </div>
  )
}

function InfoRow({
  label,
  value,
  dataTestId,
}: {
  label: string
  value: string
  dataTestId?: string
}) {
  return (
    <div>
      <div style={{ fontSize: 11, color: '#64748b' }}>{label}</div>
      <div data-testid={dataTestId} style={{ fontSize: 12, color: '#e2e8f0', marginTop: 3, wordBreak: 'break-word' }}>
        {value}
      </div>
    </div>
  )
}

function StatusPanel({
  label,
  level,
  message,
  meta,
}: {
  label: string
  level: 'healthy' | 'warning' | 'error'
  message: string
  meta: string
}) {
  const palette = level === 'healthy'
    ? { background: '#052e16', border: '#166534', text: '#bbf7d0' }
    : level === 'warning'
      ? { background: '#3b2f12', border: '#92400e', text: '#fde68a' }
      : { background: '#3f1d1d', border: '#991b1b', text: '#fecaca' }

  return (
    <div
      data-testid="graph-health-panel"
      style={{
        borderRadius: 10,
        padding: '10px 12px',
        background: palette.background,
        border: `1px solid ${palette.border}`,
      }}
    >
      <div style={{ fontSize: 11, fontWeight: 700, color: palette.text, textTransform: 'uppercase' }}>{label}</div>
      <div style={{ fontSize: 12, color: '#e2e8f0', marginTop: 4, lineHeight: 1.5 }}>{message}</div>
      <div style={{ fontSize: 11, color: '#cbd5e1', marginTop: 6, lineHeight: 1.5 }}>{meta}</div>
    </div>
  )
}

function severityColor(severity: 'error' | 'warning' | 'info') {
  if (severity === 'error') return '#ef4444'
  if (severity === 'warning') return '#f59e0b'
  return '#38bdf8'
}

function actionButtonStyle(background: string): CSSProperties {
  return {
    padding: '8px 10px',
    borderRadius: 8,
    border: 'none',
    background,
    color: '#fff',
    fontSize: 12,
    cursor: 'pointer',
  }
}
