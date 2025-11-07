import { test, expect } from "@playwright/test";

const BASE_URL = "https://rjwalters.info/thrust";

test("debug cartpole page", async ({ page }) => {
	// Capture console messages
	const consoleMessages: string[] = [];
	const errors: string[] = [];

	page.on("console", (msg) => {
		const text = `[${msg.type()}] ${msg.text()}`;
		consoleMessages.push(text);
		console.log(text);
	});

	page.on("pageerror", (error) => {
		const errorText = `[PAGE ERROR] ${error.message}`;
		errors.push(errorText);
		console.log(errorText);
	});

	console.log("\n=== Navigating to CartPole page ===");
	await page.goto(`${BASE_URL}/cartpole`, { waitUntil: "networkidle" });

	// Wait longer for WASM to initialize
	await page.waitForTimeout(10000);

	const title = await page.title();
	console.log(`\nPage title: ${title}`);

	const bodyText = await page.textContent("body");
	console.log(`\nBody text preview: ${bodyText?.substring(0, 500)}`);

	console.log("\n=== Console Messages ===");
	consoleMessages.forEach((msg) => console.log(msg));

	console.log("\n=== Errors ===");
	if (errors.length > 0) {
		errors.forEach((err) => console.log(err));
	} else {
		console.log("No errors");
	}
});

test("debug snake page", async ({ page }) => {
	// Capture console messages
	const consoleMessages: string[] = [];
	const errors: string[] = [];

	page.on("console", (msg) => {
		const text = `[${msg.type()}] ${msg.text()}`;
		consoleMessages.push(text);
		console.log(text);
	});

	page.on("pageerror", (error) => {
		const errorText = `[PAGE ERROR] ${error.message}`;
		errors.push(errorText);
		console.log(errorText);
	});

	console.log("\n=== Navigating to Snake page ===");
	await page.goto(`${BASE_URL}/snake`, { waitUntil: "networkidle" });

	// Wait longer for WASM to initialize
	await page.waitForTimeout(10000);

	const title = await page.title();
	console.log(`\nPage title: ${title}`);

	const bodyText = await page.textContent("body");
	console.log(`\nBody text preview: ${bodyText?.substring(0, 500)}`);

	console.log("\n=== Console Messages ===");
	consoleMessages.forEach((msg) => console.log(msg));

	console.log("\n=== Errors ===");
	if (errors.length > 0) {
		errors.forEach((err) => console.log(err));
	} else {
		console.log("No errors");
	}
});
