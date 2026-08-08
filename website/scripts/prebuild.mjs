#!/usr/bin/env node
// Keep the agent-readable documentation artifacts synchronized with the
// checked-in Docusaurus pages. Generation is local and deterministic: the
// script never downloads an external catalog or other network content.

import {spawnSync} from 'node:child_process';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const websiteDirectory = resolve(scriptDirectory, '..');
const generator = join(scriptDirectory, 'generate-llms-txt.py');
const result = spawnSync('python3', [generator], {
  cwd: websiteDirectory,
  stdio: 'inherit',
});

if (result.error) {
  console.error(`[prebuild] unable to run ${generator}: ${result.error.message}`);
  process.exit(1);
}

if (result.status !== 0) {
  console.error(`[prebuild] documentation generation exited with status ${result.status}`);
  process.exit(result.status ?? 1);
}
