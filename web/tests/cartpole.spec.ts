import { expect, test } from '@playwright/test';

test.describe('CartPole Game', () => {
  test('cartpole page loads correctly', async ({ page }) => {
    await page.goto('http://localhost:5173/thrust/cartpole');

    // Check page title
    await expect(page).toHaveTitle(/Thrust RL/);

    // Check main heading
    await expect(page.getByText('CartPole 3D')).toBeVisible();

    // Check back to home link
    await expect(page.getByText('← Back to Home')).toBeVisible();
  });

  test('cartpole page shows loading state', async ({ page }) => {
    await page.goto('http://localhost:5173/thrust/cartpole');

    // Initially, the page should show a loading state for WASM
    // The controls section might not be immediately visible
    await expect(page.getByText('CartPole 3D')).toBeVisible();

    // Check that we're on the right page even if WASM hasn't loaded yet
    await expect(page.getByText('← Back to Home')).toBeVisible();
  });

  test('cartpole page structure is correct', async ({ page }) => {
    await page.goto('http://localhost:5173/thrust/cartpole');

    // Check that the page has the expected layout structure
    // Even if WASM hasn't loaded, the basic structure should be there
    await expect(page.getByText('CartPole 3D')).toBeVisible();

    // Check for layout containers (these should be present regardless of WASM state)
    await expect(page.locator('.grid')).toBeVisible();
  });



  test('cartpole navigation back to home', async ({ page }) => {
    await page.goto('http://localhost:5173/thrust/cartpole');

    // Click back to home link
    await page.getByText('← Back to Home').click();

    // Should navigate back to homepage
    await expect(page.url()).toMatch(/http:\/\/localhost:5173\/thrust/);
    await expect(page.getByText('Thrust RL')).toBeVisible();
  });
});
