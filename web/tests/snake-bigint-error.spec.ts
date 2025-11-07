import { test, expect } from "@playwright/test";

const BASE_URL = "https://rjwalters.info/thrust";

test("snake should not have BigInt conversion errors", async ({ page }) => {
	const errors: string[] = [];
	const consoleErrors: string[] = [];

	// Capture all errors
	page.on("pageerror", (error) => {
		const errorText = error.message;
		errors.push(errorText);
		console.log("[PAGE ERROR]", errorText);
	});

	page.on("console", (msg) => {
		if (msg.type() === "error") {
			const text = msg.text();
			consoleErrors.push(text);
			console.log("[CONSOLE ERROR]", text);
		}
	});

	console.log("\n=== Navigating to Snake page ===");
	await page.goto(`${BASE_URL}/snake`, { waitUntil: "networkidle" });

	// Wait for WASM to load
	await page.waitForTimeout(3000);

	console.log("\n=== Clicking Start button ===");
	const startButton = page.locator('button:has-text("Start")');
	await startButton.click();

	console.log("\n=== Waiting for game to run ===");
	// Wait longer to ensure multiple steps happen
	await page.waitForTimeout(10000);

	console.log("\n=== Checking for BigInt errors ===");
	const bigIntErrors = [
		...errors.filter((e) => e.includes("BigInt")),
		...consoleErrors.filter((e) => e.includes("BigInt")),
	];

	if (bigIntErrors.length > 0) {
		console.log("Found BigInt errors:");
		bigIntErrors.forEach((err) => console.log("  -", err));
	} else {
		console.log("No BigInt errors found!");
	}

	// Assert no BigInt errors
	expect(bigIntErrors.length).toBe(0);
});
