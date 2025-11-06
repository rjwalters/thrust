import { expect, test } from '@playwright/test';

test('homepage has title', async ({ page }) => {
  await page.goto('/');

  // Expect a title "to contain" a substring.
  await expect(page).toHaveTitle(/Thrust RL/);
});

test('homepage has heading', async ({ page }) => {
  await page.goto('/');

  // Check if the page has the main heading
  await expect(page.getByText('Thrust RL')).toBeVisible();
});

test('navigation to cartpole page', async ({ page }) => {
  await page.goto('/');

  // Click on a link that navigates to cartpole
  // Adjust selector based on your actual navigation
  const cartpoleLink = page.locator('a[href="/cartpole"]').first();
  if (await cartpoleLink.isVisible()) {
    await cartpoleLink.click();
    await expect(page).toHaveURL(/.*cartpole/);
  }
});

test('navigation to snake page', async ({ page }) => {
  await page.goto('/');

  // Click on a link that navigates to snake
  // Adjust selector based on your actual navigation
  const snakeLink = page.locator('a[href="/snake"]').first();
  if (await snakeLink.isVisible()) {
    await snakeLink.click();
    await expect(page).toHaveURL(/.*snake/);
  }
});
