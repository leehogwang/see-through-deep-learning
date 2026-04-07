import fs from 'fs'
import path from 'path'
import { execFileSync } from 'child_process'
import yaml from 'yaml'

const rootDir = path.resolve(new URL('../..', import.meta.url).pathname)
const manifestPath = path.join(rootDir, 'benchmarks', 'manifest.yaml')
const manifest = yaml.parse(fs.readFileSync(manifestPath, 'utf8'))

for (const [repoId, repo] of Object.entries(manifest.repos)) {
  if (repo.url === 'local') continue

  const localDir = path.isAbsolute(repo.local_dir)
    ? repo.local_dir
    : path.join(rootDir, repo.local_dir)

  fs.mkdirSync(path.dirname(localDir), { recursive: true })

  if (!fs.existsSync(localDir)) {
    console.log(`cloning ${repoId} -> ${localDir}`)
    execFileSync('git', ['clone', '--depth', '1', '--branch', repo.ref, repo.url, localDir], { stdio: 'inherit' })
    continue
  }

  console.log(`updating ${repoId}`)
  execFileSync('git', ['fetch', '--depth', '1', 'origin', repo.ref], { cwd: localDir, stdio: 'inherit' })
  execFileSync('git', ['checkout', 'FETCH_HEAD'], { cwd: localDir, stdio: 'inherit' })
}
