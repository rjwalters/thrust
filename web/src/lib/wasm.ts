// WASM module types and loader utilities

export interface WasmCartPole {
	reset(): Float32Array;
	step(action: number): Float32Array;
	get_state(): Float32Array;
	get_episode(): number;
	get_steps(): number;
	get_best_score(): number;
}

export interface WasmSnake {
	reset(): void;
	step(actions: Int32Array): void;
	get_observation(agentId: number): Float32Array;
	num_agents(): number;
	active_agents(): Uint8Array;
	get_width(): number;
	get_height(): number;
	get_snake_positions(): Int32Array;
	get_food_positions(): Int32Array;
	get_episode(): number;
}

export interface WasmModule {
	WasmCartPole: new () => WasmCartPole;
	WasmSnake: new (
		width: number,
		height: number,
		numAgents: number,
	) => WasmSnake;
}

let wasmModule: WasmModule | null = null;
let initPromise: Promise<WasmModule> | null = null;

export async function initWasm(): Promise<WasmModule> {
	if (wasmModule) {
		console.log("[WASM] Already initialized, returning cached module");
		return wasmModule;
	}
	if (initPromise) {
		console.log("[WASM] Initialization in progress, waiting...");
		return initPromise;
	}

	console.log("[WASM] Starting initialization...");
	initPromise = (async () => {
		try {
			// Wait for WASM module to be loaded via script tag in index.html
			console.log("[WASM] Waiting for wasm_bindgen to load...");
			let attempts = 0;
			// @ts-expect-error - WASM module is in global scope
			while (!window.default && !window.wasm_bindgen) {
				attempts++;
				if (attempts % 20 === 0) {
					console.log(`[WASM] Still waiting for wasm_bindgen... (${attempts * 50}ms)`);
				}
				await new Promise((resolve) => setTimeout(resolve, 50));
			}

			console.log(`[WASM] wasm_bindgen loaded after ${attempts * 50}ms`);

			// @ts-expect-error - WASM init function from global scope
			const init = window.default || window.wasm_bindgen;
			console.log("[WASM] Calling init function...");
			await init();

			console.log("[WASM] Init complete, setting up module...");
			wasmModule = window as unknown as WasmModule;
			console.log("[WASM] Module ready!");
			return wasmModule;
		} catch (error) {
			console.error("[WASM] Failed to initialize:", error);
			throw error;
		}
	})();

	return initPromise;
}

export function getWasm(): WasmModule {
	if (!wasmModule) {
		throw new Error("WASM module not initialized. Call initWasm() first.");
	}
	return wasmModule;
}
