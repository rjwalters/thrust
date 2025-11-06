# Testing with Playwright

This project uses [Playwright](https://playwright.dev/) for end-to-end testing of the web application.

## Setup

Playwright is already configured and installed. The browsers are installed via `npx playwright install`.

## Running Tests

### All Tests
```bash
pnpm test
```

### With UI Mode (Interactive)
```bash
pnpm test:ui
```

### Specific Test File
```bash
npx playwright test tests/basic.spec.ts
```

### Debug Mode
```bash
npx playwright test --debug
```

### Generate HTML Report
```bash
npx playwright test --reporter=html
```

Then open `playwright-report/index.html` in your browser to view detailed test results.

## Configuration

The Playwright configuration is in `playwright.config.ts`. It includes:

- Tests run in parallel for speed
- Multiple browser support (Chromium, Firefox, WebKit)
- Mobile viewport testing
- Console output for local development, JSON for CI
- Automatic dev server startup (currently disabled)

## Test Structure

Tests are located in the `tests/` directory. Current tests include:

- `basic.spec.ts` - Basic functionality tests for homepage and navigation

## Writing Tests

Example test structure:

```typescript
import { expect, test } from '@playwright/test';

test('description', async ({ page }) => {
  await page.goto('/');

  // Your test logic here
  await expect(page.locator('selector')).toBeVisible();
});
```

## CI/CD

For CI environments, tests will run headlessly and with retries enabled. Make sure to set the `CI` environment variable in your CI pipeline.

## Troubleshooting

If tests fail to start the dev server, you may need to start it manually:

```bash
pnpm dev
```

Then run tests in another terminal.

## Browser Installation

If you need to reinstall browsers:

```bash
npx playwright install
```

Or for a specific browser:

```bash
npx playwright install chromium
```
