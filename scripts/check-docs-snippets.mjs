import { execFileSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const snippetDir = join(root, 'docs-snippets');
const readme = readFileSync(join(root, 'README.md'), 'utf8');
const matches = [...readme.matchAll(/<!-- docs-snippet: ([^ ]+) -->\r?\n```ts\r?\n([\s\S]*?)\r?\n```/g)];

if (matches.length === 0 || (readme.match(/```ts/g) ?? []).length !== matches.length) {
  throw new Error('Every README TypeScript block must be marked with a docs-snippet filename.');
}

const documented = new Set();
for (const match of matches) {
  const filename = match[1];
  const code = match[2];
  if (filename === undefined || code === undefined || documented.has(filename)) {
    throw new Error(`Invalid or repeated README snippet marker: ${String(filename)}`);
  }
  documented.add(filename);
  const filePath = join(snippetDir, filename);
  if (!existsSync(filePath) || readFileSync(filePath, 'utf8').trimEnd() !== code) {
    throw new Error(`README code block differs from docs-snippets/${filename}`);
  }
}

const snippetFiles = readdirSync(snippetDir).filter((name) => name.endsWith('.ts')).sort();
if (snippetFiles.length !== documented.size || snippetFiles.some((name) => !documented.has(name))) {
  throw new Error('Every docs-snippets TypeScript file must have one matching README code block.');
}
if (!existsSync(join(root, 'dist', 'index.js'))) {
  throw new Error('Run the build before checking README snippets against the packed package.');
}

const tempDir = mkdtempSync(join(tmpdir(), 'fastify-embeddings-docs-'));
try {
  const packageInfo = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));
  const tarballName = `${packageInfo.name.replaceAll('/', '-')}-${packageInfo.version}.tgz`;
  execFileSync('npm', ['pack', '--pack-destination', tempDir], {
    cwd: root,
    stdio: 'inherit',
    shell: process.platform === 'win32',
  });

  const projectDir = join(tempDir, 'project');
  mkdirSync(projectDir);
  writeFileSync(join(projectDir, 'package.json'), JSON.stringify({ private: true, type: 'module' }, null, 2));
  const tarballPath = join(tempDir, tarballName);
  execFileSync(
    'npm',
    [
      'install',
      '--prefix',
      projectDir,
      '--no-save',
      '--no-audit',
      '--no-fund',
      tarballPath,
      'fastify@^5',
      'typescript@^6',
      '@types/node@^26',
    ],
    { cwd: projectDir, stdio: 'inherit', shell: process.platform === 'win32' },
  );

  const copiedSnippets = join(projectDir, 'docs-snippets');
  cpSync(snippetDir, copiedSnippets, { recursive: true });
  writeFileSync(
    join(projectDir, 'tsconfig.json'),
    JSON.stringify(
      {
        compilerOptions: {
          target: 'ES2022',
          module: 'NodeNext',
          moduleResolution: 'NodeNext',
          strict: true,
          skipLibCheck: true,
          noEmit: true,
          types: ['node'],
        },
        include: ['docs-snippets/**/*.ts'],
      },
      null,
      2,
    ),
  );

  execFileSync(process.execPath, [join(projectDir, 'node_modules', 'typescript', 'bin', 'tsc'), '-p', join(projectDir, 'tsconfig.json')], {
    cwd: projectDir,
    stdio: 'inherit',
  });
  process.stdout.write(`README snippets match docs-snippets and typecheck against ${tarballName}.\n`);
} finally {
  // Cleanup must never fail the check. On Windows a lingering npm handle can
  // make this throw EBUSY or EPERM right after the child process exits, which
  // would otherwise turn a passing check into a red build.
  try {
    rmSync(tempDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
  } catch {
    // Ignore: the operating system reclaims its own temp directory.
  }
}
