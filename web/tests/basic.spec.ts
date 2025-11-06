import { expect, test } from '@playwright/test';

test('homepage has title', async ({ page }) => {
  await page.goto('http://localhost:5173/thrust/');

  // Expect a title "to contain" a substring.
  await expect(page).toHaveTitle(/Thrust RL/);
});

test('homepage has heading', async ({ page }) => {
  await page.goto('http://localhost:5173/thrust/');

  // Check if the page has the main heading
  await expect(page.getByText('Thrust RL')).toBeVisible();
});

test('homepage displays game cards', async ({ page }) => {
  await page.goto('http://localhost:5173/thrust/');

  // Check that both game cards are visible
  await expect(page.getByText('CartPole 3D')).toBeVisible();
  await expect(page.getByText('Multi-Agent Snake')).toBeVisible();

  // Check game descriptions
  await expect(page.getByText(/Classic control problem/)).toBeVisible();
  await expect(page.getByText(/AI agents competing in Snake/)).toBeVisible();
});

test('homepage has github link', async ({ page }) => {
  await page.goto('http://localhost:5173/thrust/');

  // Check for GitHub link
  await expect(page.getByText('View on GitHub')).toBeVisible();
});

test('navigation to cartpole page', async ({ page }) => {
  await page.goto('http://localhost:5173/thrust/');

  // Click on CartPole 3D card
  await page.getByText('CartPole 3D').click();
  await expect(page).toHaveURL('http://localhost:5173/thrust/cartpole');

  // Verify we're on the CartPole page
  await expect(page.getByText('CartPole 3D')).toBeVisible();
});

test('navigation to snake page', async ({ page }) => {
  await page.goto('http://localhost:5173/thrust/');

  // Click on Multi-Agent Snake card
  await page.getByText('Multi-Agent Snake').click();
  await expect(page).toHaveURL('http://localhost:5173/thrust/snake');

  // Verify we're on the Snake page
  await expect(page.getByText('Multi-Agent Snake')).toBeVisible();
});
