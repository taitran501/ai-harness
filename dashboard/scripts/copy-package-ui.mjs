import { readdir, mkdir, copyFile, rm, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const dashboardRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const distRoot = path.join(dashboardRoot, 'dist');
const packageWebRoot = path.resolve(dashboardRoot, '..', 'tools', 'web');
const distIndex = path.join(distRoot, 'index.html');

if (!(await stat(distIndex).catch(() => null))) {
  throw new Error('Build the dashboard before copying it into the Python package.');
}
if (path.basename(packageWebRoot) !== 'web' || path.basename(path.dirname(packageWebRoot)) !== 'tools') {
  throw new Error('Unexpected package asset destination: ' + packageWebRoot);
}

const packageAssets = path.join(packageWebRoot, 'assets');
await mkdir(packageAssets, { recursive: true });
for (const name of await readdir(packageAssets)) {
  if (/^index-[\w-]+\.(?:js|css)$/i.test(name)) {
    await rm(path.join(packageAssets, name));
  }
}

await copyFile(distIndex, path.join(packageWebRoot, 'index.html'));
const distAssets = path.join(distRoot, 'assets');
for (const name of await readdir(distAssets)) {
  const source = path.join(distAssets, name);
  const info = await stat(source);
  if (info.isFile()) await copyFile(source, path.join(packageAssets, name));
}
console.log('Copied built dashboard assets into ' + packageWebRoot);
