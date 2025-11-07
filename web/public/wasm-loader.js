// WASM loader script
// This loads the wasm-bindgen generated module and exposes it on window

import init, { WasmCartPole, WasmSnake, WasmSimpleBandit } from './pkg/thrust_rl.js';

// Initialize WASM and expose classes on window
async function loadWasm() {
	console.log('[wasm-loader] Loading WASM...');
	await init();
	console.log('[wasm-loader] WASM initialized');

	// Expose on window for the React app
	window.WasmCartPole = WasmCartPole;
	window.WasmSnake = WasmSnake;
	window.WasmSimpleBandit = WasmSimpleBandit;
	window.wasmReady = true;

	console.log('[wasm-loader] WASM classes exposed on window');
}

loadWasm().catch(err => {
	console.error('[wasm-loader] Failed to load WASM:', err);
});
