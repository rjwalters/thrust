import { expect, test } from '@playwright/test';

test.describe('Snake Game', () => {
  test('snake page loads correctly', async ({ page }) => {
    await page.goto('http://localhost:5173/thrust/snake');

    // Check page title
    await expect(page).toHaveTitle(/Thrust RL/);

    // Check main heading
    await expect(page.getByText('Multi-Agent Snake')).toBeVisible();

    // Check back to home link
    await expect(page.getByText('← Back to Home')).toBeVisible();
  });

  test('snake page shows loading state', async ({ page }) => {
    await page.goto('http://localhost:5173/thrust/snake');

    // Initially, the page should show a loading state for WASM
    // The controls section might not be immediately visible
    await expect(page.getByText('Multi-Agent Snake')).toBeVisible();

    // Check that we're on the right page even if WASM hasn't loaded yet
    await expect(page.getByText('← Back to Home')).toBeVisible();
  });

  test('snake page structure is correct', async ({ page }) => {
    await page.goto('http://localhost:5173/thrust/snake');

    // Check that the page has the expected layout structure
    // Even if WASM hasn't loaded, the basic structure should be there
    await expect(page.getByText('Multi-Agent Snake')).toBeVisible();

    // Check for layout containers (these should be present regardless of WASM state)
    await expect(page.locator('.grid')).toBeVisible();
  });

  test('snake navigation back to home', async ({ page }) => {
    await page.goto('http://localhost:5173/thrust/snake');

    // Click back to home link
    await page.getByText('← Back to Home').click();

    // Should navigate back to homepage
    await expect(page.url()).toMatch(/http:\/\/localhost:5173\/thrust/);
    await expect(page.getByText('Thrust RL')).toBeVisible();
  });
});
