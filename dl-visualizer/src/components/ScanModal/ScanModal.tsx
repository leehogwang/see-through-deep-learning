import { useEffect, useState, type ReactNode } from 'react'
import {
  scanDir,
  parseFile,
  getGitInfo,
  getBenchmarks,
  loadBenchmark,
  loadModel,
  type ScannedFile,
  type ParsedModel,
  type GitInfo,
  type LoadedModelPayload,
  type BenchmarkEntry,
} from '../../lib/api'

interface Props {
  onLoad: (payload: LoadedModelPayload) => void
  onClose: () => void
}

type Step = 'dir' | 'files' | 'models'

export default function ScanModal({ onLoad, onClose }: Props) {
  const [step, setStep] = useState<Step>('dir')
  const [dir, setDir] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [files, setFiles] = useState<ScannedFile[]>([])
  const [selectedFile, setSelectedFile] = useState<ScannedFile | null>(null)
  const [models, setModels] = useState<ParsedModel[]>([])
  const [gitInfo, setGitInfo] = useState<GitInfo>({ isGit: false })
  const [benchmarks, setBenchmarks] = useState<BenchmarkEntry[]>([])
  const [benchmarkLevel, setBenchmarkLevel] = useState<number | 'all'>('all')

  useEffect(() => {
    let cancelled = false

    getBenchmarks()
      .then((items) => {
        if (!cancelled) setBenchmarks(items)
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setError(cause instanceof Error ? cause.message : String(cause))
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  const handleScan = async () => {
    if (!dir.trim()) return
    setLoading(true)
    setError('')

    try {
      const [found, git] = await Promise.all([
        scanDir(dir.trim()),
        getGitInfo(dir.trim()),
      ])
      setFiles(found)
      setGitInfo(git)
      if (found.length === 0) {
        setError('No Python files with nn.Module found in this directory.')
      } else {
        setStep('files')
      }
    } catch (cause: unknown) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setLoading(false)
    }
  }

  const handleSelectFile = async (file: ScannedFile) => {
    setSelectedFile(file)
    setLoading(true)
    setError('')

    try {
      const parsed = await parseFile(file.file)
      setModels(parsed.models)
      setStep('models')
    } catch (cause: unknown) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setLoading(false)
    }
  }

  const handleSelectModel = (model: ParsedModel) => {
    if (!selectedFile) return
    setLoading(true)
    setError('')
    loadModel(selectedFile.file, model.name)
      .then((payload) => onLoad(payload))
      .catch((cause: unknown) => {
        setError(cause instanceof Error ? cause.message : String(cause))
      })
      .finally(() => {
        setLoading(false)
      })
  }

  const handleSelectBenchmark = async (benchmarkId: string) => {
    setLoading(true)
    setError('')

    try {
      const payload = await loadBenchmark(benchmarkId)
      onLoad(payload)
    } catch (cause: unknown) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setLoading(false)
    }
  }

  const visibleBenchmarks = benchmarks.filter((benchmark) => {
    if (benchmarkLevel === 'all') return true
    return benchmark.level === benchmarkLevel
  })

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      zIndex: 100,
      background: 'rgba(0,0,0,0.75)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: 24,
    }}>
      <div style={{
        background: '#161b27',
        border: '1px solid #2a3347',
        borderRadius: 12,
        width: '100%',
        maxWidth: step === 'dir' ? 980 : 640,
        display: 'flex',
        flexDirection: 'column',
        gap: 0,
        boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
        maxHeight: '90vh',
      }}>
        <div style={{
          padding: '16px 20px',
          borderBottom: '1px solid #2a3347',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <div>
            <p style={{ fontSize: 14, fontWeight: 600, color: '#f1f5f9' }}>
              {step === 'dir'
                ? 'Open Project or Benchmark'
                : step === 'files'
                  ? `Found ${files.length} model files`
                  : `Select model — ${selectedFile?.relative}`}
            </p>
            {step !== 'dir' && (
              <p style={{ fontSize: 11, color: '#475569', marginTop: 2 }}>
                {dir} {gitInfo.isGit ? `· git: ${gitInfo.branch}` : '· not a git repo'}
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer', fontSize: 18 }}
          >
            ✕
          </button>
        </div>

        <div style={{ padding: 20, display: 'flex', flexDirection: 'column', gap: 16, overflow: 'auto' }}>
          {step === 'dir' && (
            <>
              <section style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: '#e2e8f0' }}>Benchmark Suite</div>
                    <div style={{ fontSize: 11, color: '#64748b', marginTop: 3 }}>
                      Load the 20-model official GitHub runtime suite or the local food regression case.
                    </div>
                  </div>
                  <div style={{ flex: 1 }} />
                  <div style={{ display: 'flex', gap: 6 }}>
                    <FilterButton active={benchmarkLevel === 'all'} onClick={() => setBenchmarkLevel('all')}>
                      All
                    </FilterButton>
                    {[1, 2, 3, 4, 5].map((level) => (
                      <FilterButton
                        key={level}
                        active={benchmarkLevel === level}
                        onClick={() => setBenchmarkLevel(level)}
                      >
                        L{level}
                      </FilterButton>
                    ))}
                  </div>
                </div>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                  gap: 10,
                }}>
                  {visibleBenchmarks.map((benchmark) => (
                    <button
                      key={benchmark.id}
                      data-testid={`benchmark-card-${benchmark.id}`}
                      disabled={!benchmark.available || loading}
                      onClick={() => handleSelectBenchmark(benchmark.id)}
                      style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 8,
                        padding: '14px 16px',
                        borderRadius: 10,
                        border: `1px solid ${benchmark.available ? '#2a3347' : '#3f3f46'}`,
                        background: benchmark.available ? '#111827' : '#1f2937',
                        cursor: benchmark.available ? 'pointer' : 'default',
                        textAlign: 'left',
                        opacity: benchmark.available ? 1 : 0.6,
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{
                          fontSize: 10,
                          padding: '2px 6px',
                          borderRadius: 999,
                          background: '#1d4ed8',
                          color: '#dbeafe',
                          fontWeight: 700,
                        }}>
                          L{benchmark.level}
                        </span>
                        <span style={{ fontSize: 12, fontWeight: 600, color: '#e2e8f0' }}>
                          {benchmark.label}
                        </span>
                      </div>
                      <div style={{ fontSize: 11, color: '#94a3b8' }}>{benchmark.task}</div>
                      <div style={{ fontSize: 10, color: '#64748b', fontFamily: 'monospace' }}>
                        {benchmark.entry_file}
                      </div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                        {benchmark.tags?.map((tag) => (
                          <span
                            key={tag}
                            style={{
                              fontSize: 10,
                              padding: '2px 6px',
                              borderRadius: 999,
                              border: '1px solid #334155',
                              color: '#93c5fd',
                            }}
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    </button>
                  ))}
                </div>
              </section>

              <section style={{
                borderTop: '1px solid #243146',
                paddingTop: 16,
                display: 'flex',
                flexDirection: 'column',
                gap: 12,
              }}>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: '#e2e8f0' }}>Manual Project Load</div>
                  <div style={{ fontSize: 11, color: '#64748b', marginTop: 3 }}>
                    Enter a local repository path to scan for PyTorch models.
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 8 }}>
                  <input
                    autoFocus
                    type="text"
                    value={dir}
                    onChange={(event) => setDir(event.target.value)}
                    onKeyDown={(event) => event.key === 'Enter' && handleScan()}
                    placeholder="/home/user/my_project/VIF2"
                    style={{
                      flex: 1,
                      background: '#0f1117',
                      border: '1px solid #334155',
                      borderRadius: 6,
                      padding: '8px 12px',
                      fontSize: 12,
                      color: '#e2e8f0',
                      outline: 'none',
                      fontFamily: 'monospace',
                    }}
                  />
                  <button
                    onClick={handleScan}
                    disabled={loading}
                    style={{
                      padding: '8px 16px',
                      borderRadius: 6,
                      border: 'none',
                      background: loading ? '#374151' : '#4f46e5',
                      color: 'white',
                      fontSize: 12,
                      cursor: loading ? 'default' : 'pointer',
                    }}
                  >
                    {loading ? 'Scanning…' : 'Scan'}
                  </button>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {[
                    '/home/students/cs/202421012/food_weight_estimation/VIF2/gemini_embeddings',
                    '/home/students/cs/202421012/food_weight_estimation/VIF2',
                  ].map((shortcut) => (
                    <button
                      key={shortcut}
                      onClick={() => setDir(shortcut)}
                      style={{
                        fontSize: 10,
                        padding: '3px 8px',
                        borderRadius: 4,
                        border: '1px solid #334155',
                        background: '#1e2535',
                        color: '#94a3b8',
                        cursor: 'pointer',
                        fontFamily: 'monospace',
                      }}
                    >
                      {shortcut.split('/').slice(-2).join('/')}
                    </button>
                  ))}
                </div>
              </section>
            </>
          )}

          {step === 'files' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 360, overflowY: 'auto' }}>
              {files.map((file) => (
                <button
                  key={file.file}
                  onClick={() => handleSelectFile(file)}
                  disabled={loading}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '10px 12px',
                    borderRadius: 8,
                    border: '1px solid #2a3347',
                    background: '#1a2035',
                    cursor: 'pointer',
                    textAlign: 'left',
                  }}
                >
                  <div>
                    <p style={{ fontSize: 12, fontWeight: 500, color: '#e2e8f0', fontFamily: 'monospace' }}>
                      {file.relative}
                    </p>
                    <p style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>
                      {file.models.join(', ')}
                    </p>
                  </div>
                  <span style={{ fontSize: 11, color: '#6366f1', marginLeft: 12 }}>
                    {file.models.length} model{file.models.length > 1 ? 's' : ''} →
                  </span>
                </button>
              ))}
            </div>
          )}

          {step === 'models' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 360, overflowY: 'auto' }}>
              {models.length === 0 && (
                <p style={{ fontSize: 12, color: '#64748b' }}>
                  No parseable models found. The file may rely on custom base classes or unsupported syntax.
                </p>
              )}
              {models.map((model) => (
                <button
                  key={model.name}
                  onClick={() => handleSelectModel(model)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '12px 14px',
                    borderRadius: 8,
                    border: '1px solid #2a3347',
                    background: '#1a2035',
                    cursor: 'pointer',
                    textAlign: 'left',
                  }}
                >
                  <div>
                    <p style={{ fontSize: 13, fontWeight: 600, color: '#a5b4fc' }}>{model.name}</p>
                    <p style={{ fontSize: 11, color: '#64748b', marginTop: 3 }}>
                      {model.layers.length > 0
                        ? model.layers.map((layer) => layer.label).join(' → ')
                        : 'No directly parseable layers (uses sub-modules)'}
                    </p>
                  </div>
                  <span style={{ fontSize: 20, color: '#4f46e5', marginLeft: 8 }}>→</span>
                </button>
              ))}
            </div>
          )}

          {error && (
            <p style={{
              fontSize: 12,
              color: '#f87171',
              padding: '8px 12px',
              background: '#1f0a0a',
              borderRadius: 6,
            }}>
              {error}
            </p>
          )}
        </div>

        {step !== 'dir' && (
          <div style={{ padding: '12px 20px', borderTop: '1px solid #2a3347', display: 'flex', gap: 8 }}>
            <button
              onClick={() => {
                setStep(step === 'models' ? 'files' : 'dir')
                setError('')
              }}
              style={{
                padding: '6px 14px',
                borderRadius: 6,
                border: '1px solid #334155',
                background: 'transparent',
                color: '#94a3b8',
                fontSize: 12,
                cursor: 'pointer',
              }}
            >
              ← Back
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

function FilterButton({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: ReactNode
}) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '5px 10px',
        borderRadius: 999,
        border: `1px solid ${active ? '#1d4ed8' : '#334155'}`,
        background: active ? '#172554' : '#111827',
        color: active ? '#bfdbfe' : '#94a3b8',
        fontSize: 11,
        cursor: 'pointer',
      }}
    >
      {children}
    </button>
  )
}
