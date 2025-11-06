/// <reference types="vite/client" />

// Build-time constants injected by Vite
declare const __COMMIT_HASH__: string;
declare const __BUILD_TIME__: string;

// WASM module types loaded by wasm-loader.js
interface Window {
	wasmReady?: boolean;
	WasmCartPole?: any;
	WasmSnake?: any;
}
