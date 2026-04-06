import express from 'express'
import { execSync, execFileSync } from 'child_process'
import { fileURLToPath } from 'url'
import path from 'path'
import fs from 'fs'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const app = express()
const PORT = 5173

app.use(express.json())
app.use(express.static(path.join(__dirname, 'dist')))

// ── Scan directory for Python model files ───────────────────
app.get('/api/scan', (req, res) => {
  const dir = req.query.dir
  if (!dir) return res.status(400).json({ error: 'dir required' })
  if (!fs.existsSync(dir)) return res.status(404).json({ error: `Directory not found: ${dir}` })

  try {
    const result = execFileSync('python3', [
      path.join(__dirname, 'parse_model.py'), 'scan', dir
    ], { timeout: 15000 })
    res.json(JSON.parse(result.toString()))
  } catch (e) {
    res.status(500).json({ error: e.message })
  }
})

// ── Parse a specific Python file ─────────────────────────────
app.get('/api/parse', (req, res) => {
  const file = req.query.file
  if (!file) return res.status(400).json({ error: 'file required' })
  if (!fs.existsSync(file)) return res.status(404).json({ error: `File not found: ${file}` })

  try {
    const result = execFileSync('python3', [
      path.join(__dirname, 'parse_model.py'), 'parse', file
    ], { timeout: 10000 })
    res.json(JSON.parse(result.toString()))
  } catch (e) {
    res.status(500).json({ error: e.message })
  }
})

// ── Parse entire directory → global model registry ───────────
// Returns { modelName -> ParsedModel } across all .py files in dir
app.get('/api/parse-dir', (req, res) => {
  const dir = req.query.dir
  if (!dir) return res.status(400).json({ error: 'dir required' })
  if (!fs.existsSync(dir)) return res.status(404).json({ error: `Not found: ${dir}` })

  try {
    // First scan to get all model files
    const scanResult = execFileSync('python3', [
      path.join(__dirname, 'parse_model.py'), 'scan', dir
    ], { timeout: 15000 })
    const files = JSON.parse(scanResult.toString())

    // Parse each file and merge into registry
    const registry = {}
    for (const f of files) {
      try {
        const parsed = execFileSync('python3', [
          path.join(__dirname, 'parse_model.py'), 'parse', f.file
        ], { timeout: 8000 })
        const { models } = JSON.parse(parsed.toString())
        for (const m of (models || [])) {
          if (!(m.name in registry)) {   // first definition wins
            registry[m.name] = m
          }
        }
      } catch { /* skip unparseable files */ }
    }

    res.json(registry)
  } catch (e) {
    res.status(500).json({ error: e.message })
  }
})

// ── Git info for a path ───────────────────────────────────────
app.get('/api/git-info', (req, res) => {
  const dir = req.query.dir
  if (!dir) return res.status(400).json({ error: 'dir required' })

  try {
    const root = execSync('git rev-parse --show-toplevel', {
      cwd: dir, timeout: 5000
    }).toString().trim()
    const branch = execSync('git rev-parse --abbrev-ref HEAD', {
      cwd: dir, timeout: 5000
    }).toString().trim()
    res.json({ root, branch, isGit: true })
  } catch {
    res.json({ isGit: false })
  }
})

// ── Create git worktree ───────────────────────────────────────
app.post('/api/worktree', (req, res) => {
  const { repoRoot, sourceFile } = req.body
  if (!repoRoot) return res.status(400).json({ error: 'repoRoot required' })

  const ts = Date.now()
  const branch = `dl-viz-edit-${ts}`
  const wtPath = `/tmp/dl-worktree-${ts}`

  try {
    execSync(`git worktree add "${wtPath}" -b "${branch}"`, {
      cwd: repoRoot, timeout: 15000
    })
    res.json({ worktreePath: wtPath, branch, success: true })
  } catch (e) {
    res.status(500).json({ error: e.message })
  }
})

// ── Save modified code to worktree ───────────────────────────
app.post('/api/save', (req, res) => {
  const { worktreePath, repoRoot, sourceFile, content } = req.body
  if (!worktreePath || !sourceFile || content === undefined) {
    return res.status(400).json({ error: 'worktreePath, sourceFile, content required' })
  }

  try {
    // Compute relative path from repo root
    const relPath = path.relative(repoRoot, sourceFile)
    const targetPath = path.join(worktreePath, relPath)

    // Ensure directory exists
    fs.mkdirSync(path.dirname(targetPath), { recursive: true })
    fs.writeFileSync(targetPath, content, 'utf-8')

    // Git add + commit
    execSync(`git add "${relPath}" && git commit -m "dl-viz: edited ${path.basename(sourceFile)}"`, {
      cwd: worktreePath, timeout: 10000
    })

    res.json({ success: true, savedTo: targetPath })
  } catch (e) {
    res.status(500).json({ error: e.message })
  }
})

// ── Read file content ─────────────────────────────────────────
app.get('/api/read', (req, res) => {
  const file = req.query.file
  if (!file || !fs.existsSync(file)) return res.status(404).json({ error: 'not found' })
  res.json({ content: fs.readFileSync(file, 'utf-8') })
})

// Fallback → SPA
app.get('/{*path}', (_, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'))
})

app.listen(PORT, () => {
  console.log(`\n  ➜  DL Visualizer  http://localhost:${PORT}\n`)
})
