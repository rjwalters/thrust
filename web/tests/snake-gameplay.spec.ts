import { test } from "@playwright/test";

const BASE_URL = "https://rjwalters.info/thrust";

test("snake gameplay with step calls", async ({ page }) => {
	// Capture console messages and errors
	const consoleMessages: string[] = [];
	const errors: string[] = [];

	page.on("console", (msg) => {
		const text = `[${msg.type()}] ${msg.text()}`;
		consoleMessages.push(text);
		console.log(text);
	});

	page.on("pageerror", (error) => {
		const errorText = `[PAGE ERROR] ${error.message}\n${error.stack}`;
		errors.push(errorText);
		console.log(errorText);
	});

	console.log("\n=== Navigating to Snake page ===");
	await page.goto(`${BASE_URL}/snake`, { waitUntil: "networkidle" });

	// Wait for WASM to load
	await page.waitForTimeout(3000);

	console.log("\n=== Clicking Start button ===");
	// Find and click the Start button
	const startButton = page.locator('button:has-text("Start")');
	await startButton.click();

	console.log("\n=== Waiting for game to run ===");
	// Wait for game loop to run and potentially error
	await page.waitForTimeout(5000);

	console.log("\n=== Final Console Messages ===");
	consoleMessages.forEach((msg) => console.log(msg));

	console.log("\n=== Errors ===");
	if (errors.length > 0) {
		errors.forEach((err) => console.log(err));
	} else {
		console.log("No errors");
	}
});
