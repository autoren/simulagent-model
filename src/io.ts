import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

export async function readJson<T>(path: string): Promise<T> {
  return JSON.parse(await readFile(path, 'utf8')) as T;
}

export async function readJsonl<T>(path: string): Promise<T[]> {
  const content = await readFile(path, 'utf8');
  return content
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line) as T);
}

export async function writeJson(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

export async function writeJsonl(path: string, values: unknown[]): Promise<string> {
  await mkdir(dirname(path), { recursive: true });
  const content = values.map((value) => JSON.stringify(value)).join('\n') + '\n';
  await writeFile(path, content, 'utf8');
  return content;
}
