// Turns docs/*.html (written by screens.py) into PNGs at 2x.
import { createRequire } from 'node:module';
import { readdirSync, unlinkSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const { chromium } = createRequire(import.meta.url)('playwright');
const docs = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'docs');
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 900, height: 400 }, deviceScaleFactor: 2 });
for (const f of readdirSync(docs).filter((f) => f.endsWith('.html'))) {
  await page.goto('file://' + join(docs, f));
  await page.locator('pre').screenshot({ path: join(docs, f.replace('.html', '.png')) });
  unlinkSync(join(docs, f));
  console.log('rendered', f.replace('.html', '.png'));
}
await browser.close();
