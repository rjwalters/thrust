import { test, expect } from '@playwright/test';

test('production site loads', async ({ page }) => {
  const errors: string[] = [];
  const logs: string[] = [];

  // Capture console messages before navigation
  page.on('console', msg => {
    const text = msg.text();
    logs.push(`[${msg.type()}] ${text}`);
    if (msg.type() === 'error') {
      errors.push(text);
    }
  });

  // Capture page errors
  page.on('pageerror', error => {
    errors.push(`Page error: ${error.message}`);
  });

  // Go to the deployed site
  console.log('Navigating to https://rjwalters.github.io/thrust/');
  await page.goto('https://rjwalters.github.io/thrust/', { waitUntil: 'networkidle' });

  // Check if the title is correct
  const title = await page.title();
  console.log('Page title:', title);

  // Check for the root div
  const root = page.locator('#root');
  const rootExists = await root.count() > 0;
  console.log('Root div exists:', rootExists);

  // Wait a bit to see if React app renders
  await page.waitForTimeout(3000);

  // Get the rendered content of #root
  const rootContent = await root.innerHTML();
  console.log('Root innerHTML length:', rootContent.length);
  console.log('Root content preview:', rootContent.substring(0, 500));

  // Get the full page text
  const bodyText = await page.locator('body').innerText();
  console.log('Body text:', bodyText);

  // Log all console messages
  console.log('\n=== Console Logs ===');
  logs.forEach(log => console.log(log));

  // Log any errors
  if (errors.length > 0) {
    console.log('\n=== Errors Found ===');
    errors.forEach(err => console.log(err));
    throw new Error(`Found ${errors.length} error(s) on page`);
  }
});
