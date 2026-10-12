/**
 * Verify that the distribution build contains no unresolved #src path aliases.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

const distDir = join(process.cwd(), 'dist');

function checkDir(dir) {
  const entries = readdirSync(dir);
  let failed = false;

  for (const entry of entries) {
    const fullPath = join(dir, entry);
    const stat = statSync(fullPath);

    if (stat.isDirectory()) {
      if (checkDir(fullPath)) failed = true;
    } else if (/\.(js|cjs|mjs|d\.ts|d\.cts)$/.test(entry)) {
      const content = readFileSync(fullPath, 'utf8');
      if (content.includes('#src')) {
        console.error(`Error: Unresolved alias '#src' found in ${fullPath}`);
        failed = true;
      }
    }
  }

  return failed;
}

const failed = checkDir(distDir);
if (failed) {
  process.exit(1);
} else {
  console.log('Build verification passed: dist is clean of #src aliases.');
}
