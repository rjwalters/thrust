import { test, expect } from "@playwright/test";

const BASE_URL = "https://rjwalters.info/thrust";

// The training dashboard is a new, static (no-WASM) page. It is tested against
// the local dev server (started by playwright.config.ts `webServer`) rather than
// the production URL above, since it is not deployed until this change merges.
const LOCAL_URL = "http://localhost:5173/thrust";

test("home page loads", async ({ page }) => {
	const errors: string[] = [];

	page.on("pageerror", (error) => {
		errors.push(error.message);
		console.log("[PAGE ERROR]", error.message);
	});

	console.log("\n=== Navigating to home page ===");
	await page.goto(BASE_URL, { waitUntil: "networkidle" });

	// Check if page loaded
	const title = await page.textContent("h1");
	console.log("Page title:", title);

	// Check for errors
	if (errors.length > 0) {
		console.log("Errors found:");
		errors.forEach((err) => console.log("  -", err));
	}

	expect(title).toContain("Thrust");
	expect(errors.length).toBe(0);
});

test("cartpole page loads", async ({ page }) => {
	const errors: string[] = [];

	page.on("pageerror", (error) => {
		errors.push(error.message);
		console.log("[PAGE ERROR]", error.message);
	});

	console.log("\n=== Navigating to CartPole page ===");
	await page.goto(`${BASE_URL}/cartpole`, { waitUntil: "networkidle" });

	// Wait for WASM
	await page.waitForTimeout(3000);

	const title = await page.textContent("h1");
	console.log("Page title:", title);

	if (errors.length > 0) {
		console.log("Errors found:");
		errors.forEach((err) => console.log("  -", err));
	}

	expect(title).toContain("CartPole");
	expect(errors.length).toBe(0);
});

test("snake page loads", async ({ page }) => {
	const errors: string[] = [];

	page.on("pageerror", (error) => {
		errors.push(error.message);
		console.log("[PAGE ERROR]", error.message);
	});

	console.log("\n=== Navigating to Snake page ===");
	await page.goto(`${BASE_URL}/snake`, { waitUntil: "networkidle" });

	// Wait for WASM
	await page.waitForTimeout(3000);

	const title = await page.textContent("h1");
	console.log("Page title:", title);

	if (errors.length > 0) {
		console.log("Errors found:");
		errors.forEach((err) => console.log("  -", err));
	}

	expect(title).toContain("Snake");
	expect(errors.length).toBe(0);
});

test("training dashboard page loads", async ({ page }) => {
	const errors: string[] = [];

	page.on("pageerror", (error) => {
		errors.push(error.message);
		console.log("[PAGE ERROR]", error.message);
	});

	console.log("\n=== Navigating to Training Dashboard page ===");
	await page.goto(`${LOCAL_URL}/dashboard`, { waitUntil: "networkidle" });

	const title = await page.textContent("h1");
	console.log("Page title:", title);

	// The CSV curves should load and render an SVG chart (recharts).
	await expect(page.locator(".recharts-surface").first()).toBeVisible();

	if (errors.length > 0) {
		console.log("Errors found:");
		errors.forEach((err) => console.log("  -", err));
	}

	expect(title).toContain("Training");
	expect(errors.length).toBe(0);
});
