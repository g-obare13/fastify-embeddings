/**
 * Verify project conventions across all source files in `src/`:
 * - No em dashes (— or –)
 * - No TODOs
 * - No console.log
 * - No TypeScript any type (: any, as any, <any>)
 * - Top-level import type only
 * - Same-folder imports use './x.js', everything else uses '#src/...'
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';

const srcDir = join(process.cwd(), 'src');
let violations = 0;

function report(file, lineNo, rule, message) {
  console.error(`Convention violation: [${rule}] ${file}:${lineNo} - ${message}`);
  violations++;
}

function checkFile(file) {
  const content = readFileSync(file, 'utf8');
  const lines = content.split('\n');
  const fileDir = dirname(file);

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const lineNo = i + 1;

    // 1. Em dash check
    if (/[\u2014\u2013]/.test(line)) {
      report(file, lineNo, 'no-em-dash', 'Found em dash (— or –)');
    }

    // 2. TODO check
    if (/\bTODO\b/i.test(line)) {
      report(file, lineNo, 'no-todo', 'Found TODO');
    }

    // 3. console.log check
    if (/console\.log/.test(line)) {
      report(file, lineNo, 'no-console-log', 'Found console.log');
    }

    // 4. any type check (: any, as any, <any>, Promise<any>)
    if (/:\s*any\b|\bas\s+any\b|<any>|Promise<any>/.test(line)) {
      report(file, lineNo, 'no-any', 'Found TypeScript any type');
    }

    // 5. top-level import type only (no inline `import { type Foo }`)
    if (/import\s*\{[^}]*\btype\s+\w+/.test(line)) {
      report(file, lineNo, 'top-level-import-type', 'Found inline `type` specifier in import. Use top-level `import type`');
    }

    // 6. import specifier conventions:
    // If importing from same folder, must use `./x.js`. If from different folder within project, must use `#src/...`
    const importMatch = line.match(/(?:import|export)\s+.*?\s+from\s+['"]([^'"]+)['"]/);
    if (importMatch) {
      const spec = importMatch[1];
      if (spec.startsWith('.')) {
        // Relative import: must only target same directory, no '../'
        if (spec.startsWith('../')) {
          report(file, lineNo, 'import-convention', `Relative import '${spec}' traverses parent directory. Use '#src/...' instead.`);
        }
      } else if (spec.startsWith('#src/')) {
        // Check if this could have been a same-folder import
        const targetRel = spec.replace('#src/', '');
        const targetPath = join(srcDir, targetRel.replace(/\.js$/, '.ts'));
        if (dirname(targetPath) === fileDir) {
          report(file, lineNo, 'import-convention', `Import '${spec}' targets same folder. Use './${targetRel.split('/').pop()}' instead.`);
        }
      }
    }
  }
}

function traverse(dir) {
  const entries = readdirSync(dir);
  for (const entry of entries) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      traverse(full);
    } else if (/\.ts$/.test(entry)) {
      checkFile(full);
    }
  }
}

traverse(srcDir);

if (violations > 0) {
  console.error(`Total violations found: ${violations}`);
  process.exit(1);
} else {
  console.log('Convention check passed: all source files comply.');
}
