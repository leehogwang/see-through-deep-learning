import { useState } from 'react'
import { scanDir, parseFile, parseDirRegistry, getGitInfo, type ScannedFile, type ParsedModel, type GitInfo } from '../../lib/api'

interface Props {
  onLoad: (model: ParsedModel, sourceFile: string, gitInfo: GitInfo, registry: Record<string, ParsedModel>) => void
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
  const [registry, setRegistry] = useState<Record<string, ParsedModel>>({})

  const handleScan = async () => {
    if (!dir.trim()) return
    setLoading(true); setError('')
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
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  const handleSelectFile = async (f: ScannedFile) => {
    setSelectedFile(f)
    setLoading(true); setError('')
    try {
      // Parse the selected file AND build a full directory registry in parallel
      const fileDir = f.file.substring(0, f.file.lastIndexOf('/'))
      const [parsed, reg] = await Promise.all([
        parseFile(f.file),
        parseDirRegistry(fileDir).catch(() => ({} as Record<string, ParsedModel>)),
      ])
      setModels(parsed.models)
      setRegistry(reg)
      setStep('models')
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  const handleSelectModel = (m: ParsedModel) => {
    if (!selectedFile) return
    onLoad(m, selectedFile.file, gitInfo, registry)
  }

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 100,
      background: 'rgba(0,0,0,0.75)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: 24,
    }}>
      <div style={{
        background: '#161b27', border: '1px solid #2a3347',
        borderRadius: 12, width: '100%', maxWidth: 560,
        display: 'flex', flexDirection: 'column', gap: 0,
        boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
      }}>
        {/* Header */}
        <div style={{ padding: '16px 20px', borderBottom: '1px solid #2a3347', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <p style={{ fontSize: 14, fontWeight: 600, color: '#f1f5f9' }}>
              {step === 'dir' ? 'Open Project Directory' :
               step === 'files' ? `Found ${files.length} model files` :
               `Select model — ${selectedFile?.relative}`}
            </p>
            {step !== 'dir' && (
              <p style={{ fontSize: 11, color: '#475569', marginTop: 2 }}>
                {dir} {gitInfo.isGit ? `· git: ${gitInfo.branch}` : '· not a git repo'}
              </p>
            )}
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer', fontSize: 18 }}>✕</button>
        </div>

        {/* Body */}
        <div style={{ padding: 20, display: 'flex', flexDirection: 'column', gap: 12 }}>

          {/* Step: Directory input */}
          {step === 'dir' && (
            <>
              <p style={{ fontSize: 12, color: '#64748b' }}>
                Enter the path to your project directory. All Python files containing <code style={{ color: '#818cf8' }}>nn.Module</code> subclasses will be listed.
              </p>
              <div style={{ display: 'flex', gap: 8 }}>
                <input
                  autoFocus
                  type="text"
                  value={dir}
                  onChange={e => setDir(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleScan()}
                  placeholder="/home/user/my_project/VIF2"
                  style={{
                    flex: 1, background: '#0f1117', border: '1px solid #334155',
                    borderRadius: 6, padding: '8px 12px', fontSize: 12,
                    color: '#e2e8f0', outline: 'none', fontFamily: 'monospace',
                  }}
                />
                <button
                  onClick={handleScan}
                  disabled={loading}
                  style={{
                    padding: '8px 16px', borderRadius: 6, border: 'none',
                    background: loading ? '#374151' : '#4f46e5',
                    color: 'white', fontSize: 12, cursor: loading ? 'default' : 'pointer',
                  }}
                >
                  {loading ? 'Scanning…' : 'Scan'}
                </button>
              </div>
              {/* Quick shortcuts */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {[
                  '/home/students/cs/202421012/food_weight_estimation/VIF2/gemini_embeddings',
                  '/home/students/cs/202421012/food_weight_estimation/VIF2',
                ].map(p => (
                  <button key={p} onClick={() => setDir(p)} style={{
                    fontSize: 10, padding: '3px 8px', borderRadius: 4,
                    border: '1px solid #334155', background: '#1e2535',
                    color: '#94a3b8', cursor: 'pointer', fontFamily: 'monospace',
                  }}>
                    {p.split('/').slice(-2).join('/')}
                  </button>
                ))}
              </div>
            </>
          )}

          {/* Step: File list */}
          {step === 'files' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 360, overflowY: 'auto' }}>
              {files.map(f => (
                <button key={f.file} onClick={() => handleSelectFile(f)}
                  disabled={loading}
                  style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    padding: '10px 12px', borderRadius: 8, border: '1px solid #2a3347',
                    background: '#1a2035', cursor: 'pointer', textAlign: 'left',
                  }}
                  onMouseEnter={e => (e.currentTarget.style.borderColor = '#6366f1')}
                  onMouseLeave={e => (e.currentTarget.style.borderColor = '#2a3347')}
                >
                  <div>
                    <p style={{ fontSize: 12, fontWeight: 500, color: '#e2e8f0', fontFamily: 'monospace' }}>{f.relative}</p>
                    <p style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>
                      {f.models.join(', ')}
                    </p>
                  </div>
                  <span style={{ fontSize: 11, color: '#6366f1', marginLeft: 12 }}>{f.models.length} model{f.models.length > 1 ? 's' : ''} →</span>
                </button>
              ))}
            </div>
          )}

          {/* Step: Model list */}
          {step === 'models' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 360, overflowY: 'auto' }}>
              {models.length === 0 && (
                <p style={{ fontSize: 12, color: '#64748b' }}>No parseable models found (may use custom base classes).</p>
              )}
              {models.map(m => (
                <button key={m.name} onClick={() => handleSelectModel(m)}
                  style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    padding: '12px 14px', borderRadius: 8, border: '1px solid #2a3347',
                    background: '#1a2035', cursor: 'pointer', textAlign: 'left',
                  }}
                  onMouseEnter={e => (e.currentTarget.style.borderColor = '#6366f1')}
                  onMouseLeave={e => (e.currentTarget.style.borderColor = '#2a3347')}
                >
                  <div>
                    <p style={{ fontSize: 13, fontWeight: 600, color: '#a5b4fc' }}>{m.name}</p>
                    <p style={{ fontSize: 11, color: '#64748b', marginTop: 3 }}>
                      {m.layers.length > 0
                        ? m.layers.map(l => l.label).join(' → ')
                        : 'No directly parseable layers (uses sub-modules)'}
                    </p>
                  </div>
                  <span style={{ fontSize: 20, color: '#4f46e5', marginLeft: 8 }}>→</span>
                </button>
              ))}
            </div>
          )}

          {error && (
            <p style={{ fontSize: 12, color: '#f87171', padding: '8px 12px', background: '#1f0a0a', borderRadius: 6 }}>{error}</p>
          )}
        </div>

        {/* Footer nav */}
        {step !== 'dir' && (
          <div style={{ padding: '12px 20px', borderTop: '1px solid #2a3347', display: 'flex', gap: 8 }}>
            <button
              onClick={() => { setStep(step === 'models' ? 'files' : 'dir'); setError('') }}
              style={{ padding: '6px 14px', borderRadius: 6, border: '1px solid #334155', background: 'transparent', color: '#94a3b8', fontSize: 12, cursor: 'pointer' }}
            >
              ← Back
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
