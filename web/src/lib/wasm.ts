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
	if (wasmModule) return wasmModule;
	if (initPromise) return initPromise;

	initPromise = (async () => {
		try {
			// Wait for WASM module to be loaded via script tag in index.html
			// @ts-expect-error - WASM module is in global scope
			while (!window.default && !window.wasm_bindgen) {
				await new Promise((resolve) => setTimeout(resolve, 50));
			}

			// @ts-expect-error - WASM init function from global scope
			const init = window.default || window.wasm_bindgen;
			await init();

			wasmModule = window as unknown as WasmModule;
			return wasmModule;
		} catch (error) {
			console.error("Failed to initialize WASM:", error);
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
