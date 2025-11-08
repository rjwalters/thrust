import { useEffect, useRef, useState } from "react";
import type { WasmSnake } from "../../lib/wasm";
import { initWasm } from "../../lib/wasm";

export interface SnakeState {
	width: number;
	height: number;
	numAgents: number;
	episode: number;
	steps: number;
	snakePositions: Int32Array;
	foodPositions: Int32Array;
	activeAgents: Uint8Array;
}

export interface UseSnakeResult {
	state: SnakeState | null;
	isRunning: boolean;
	isPaused: boolean;
	speed: number;
	actualFps: number;
	loadingProgress: number;
	loadingStatus: string;
	start: () => void;
	pause: () => void;
	reset: () => void;
	setSpeed: (speed: number) => void;
}

const GRID_WIDTH = 20;
const GRID_HEIGHT = 20;
const NUM_AGENTS = 4; // Four snakes competing

export function useSnake(): UseSnakeResult {
	const [state, setState] = useState<SnakeState | null>(null);
	const [isRunning, setIsRunning] = useState(false);
	const [isPaused, setIsPaused] = useState(false);
	const [speed, setSpeed] = useState(1);
	const [actualFps, setActualFps] = useState(0);
	const [loadingProgress, setLoadingProgress] = useState(0);
	const [loadingStatus, setLoadingStatus] = useState("Initializing WASM...");

	const envRef = useRef<WasmSnake | null>(null);
	const frameIdRef = useRef<number | null>(null);
	const lastFrameTimeRef = useRef<number>(0);
	const stepsRef = useRef<number>(0);
	const fpsFrameTimesRef = useRef<number[]>([]);

	// Initialize WASM environment
	useEffect(() => {
		let mounted = true;

		async function init() {
			try {
				const wasm = await initWasm();
				if (mounted) {
					envRef.current = new wasm.WasmSnake(
						GRID_WIDTH,
						GRID_HEIGHT,
						NUM_AGENTS,
					);
					envRef.current.reset();

					// Load policy from JSON with progress tracking
					try {
						console.log("[Snake] Loading policy model...");
						setLoadingStatus("Downloading AI model (125MB)...");

						const response = await fetch("https://thrust-models.personal-account-251.workers.dev/snake_model.json");
						if (!response.ok) {
							console.warn("[Snake] Policy model not found, using random actions");
							setLoadingStatus("Model not found - using random actions");
							return;
						}

						const contentLength = response.headers.get('content-length');
						const total = contentLength ? parseInt(contentLength, 10) : 131072000; // ~125MB fallback

						let loaded = 0;
						const reader = response.body?.getReader();
						const chunks: BlobPart[] = [];

						if (reader) {
							while (true) {
								const { done, value } = await reader.read();
								if (done) break;

								chunks.push(value);
								loaded += value.length;
								const progress = Math.round((loaded / total) * 100);
								const loadedMB = (loaded / 1024 / 1024).toFixed(1);
								const totalMB = (total / 1024 / 1024).toFixed(1);
								setLoadingProgress(progress);
								setLoadingStatus(`Downloading AI model: ${loadedMB}MB / ${totalMB}MB (${progress}%)`);
							}

							// Combine chunks and decode
							const blob = new Blob(chunks);
							const policyJson = await blob.text();

							setLoadingStatus("Loading model into memory...");
							console.log(`[Snake] Policy JSON loaded (${policyJson.length} bytes)`);
							envRef.current.load_policy_json(policyJson);
							console.log("[Snake] Policy loaded successfully");
							setLoadingStatus("Ready");
							setLoadingProgress(100);
						}
					} catch (error) {
						console.warn("[Snake] Failed to load policy:", error);
						setLoadingStatus("Failed to load model - using random actions");
					}

					// Initialize state
					setState({
						width: GRID_WIDTH,
						height: GRID_HEIGHT,
						numAgents: NUM_AGENTS,
						episode: envRef.current.get_episode(),
						steps: 0,
						snakePositions: envRef.current.get_snake_positions(),
						foodPositions: envRef.current.get_food_positions(),
						activeAgents: envRef.current.active_agents(),
					});
				}
			} catch (error) {
				console.error("Failed to initialize Snake WASM:", error);
			}
		}

		init();

		return () => {
			mounted = false;
			if (frameIdRef.current !== null) {
				cancelAnimationFrame(frameIdRef.current);
			}
		};
	}, []);

	// Game loop
	useEffect(() => {
		if (!isRunning || isPaused || !envRef.current) {
			return;
		}

		const targetFrameTime = 100 / speed; // Base: 10 FPS, scales with speed

		function gameLoop(currentTime: number) {
			if (!envRef.current || !isRunning || isPaused) return;

			const elapsed = currentTime - lastFrameTimeRef.current;

			// Calculate actual FPS (smoothed over last 30 frames)
			fpsFrameTimesRef.current.push(currentTime);
			if (fpsFrameTimesRef.current.length > 30) {
				fpsFrameTimesRef.current.shift();
			}
			if (fpsFrameTimesRef.current.length >= 2) {
				const timeSpan = currentTime - fpsFrameTimesRef.current[0];
				const frameCount = fpsFrameTimesRef.current.length - 1;
				const fps = (frameCount / timeSpan) * 1000;
				setActualFps(Math.round(fps));
			}

			if (elapsed >= targetFrameTime) {
				lastFrameTimeRef.current = currentTime;

				// Generate actions for all agents
				// 0: up, 1: down, 2: left, 3: right
				const actions = new Int32Array(NUM_AGENTS);
				for (let i = 0; i < NUM_AGENTS; i++) {
					if (envRef.current.has_policy()) {
						// Use trained policy
						const action = envRef.current.get_policy_action(i);
						actions[i] = action >= 0 ? action : Math.floor(Math.random() * 4);
					} else {
						// Fall back to random actions if no policy loaded
						actions[i] = Math.floor(Math.random() * 4);
					}
				}

				// Step environment with all actions
				envRef.current.step(actions);
				stepsRef.current++;

				// Check if all agents are dead
				const activeAgents = envRef.current.active_agents();
				const allDead = Array.from(activeAgents).every((active) => !active);

				if (allDead) {
					// Auto-reset for continuous demo
					envRef.current.reset();
					stepsRef.current = 0;
				}

				// Update state
				setState({
					width: GRID_WIDTH,
					height: GRID_HEIGHT,
					numAgents: NUM_AGENTS,
					episode: envRef.current.get_episode(),
					steps: stepsRef.current,
					snakePositions: envRef.current.get_snake_positions(),
					foodPositions: envRef.current.get_food_positions(),
					activeAgents: envRef.current.active_agents(),
				});
			}

			frameIdRef.current = requestAnimationFrame(gameLoop);
		}

		lastFrameTimeRef.current = performance.now();
		frameIdRef.current = requestAnimationFrame(gameLoop);

		return () => {
			if (frameIdRef.current !== null) {
				cancelAnimationFrame(frameIdRef.current);
			}
		};
	}, [isRunning, isPaused, speed]);

	const start = () => {
		if (!isRunning && envRef.current) {
			setIsRunning(true);
			setIsPaused(false);
		}
	};

	const pause = () => {
		setIsPaused(!isPaused);
	};

	const reset = () => {
		if (envRef.current) {
			envRef.current.reset();
			stepsRef.current = 0;
			setState({
				width: GRID_WIDTH,
				height: GRID_HEIGHT,
				numAgents: NUM_AGENTS,
				episode: envRef.current.get_episode(),
				steps: 0,
				snakePositions: envRef.current.get_snake_positions(),
				foodPositions: envRef.current.get_food_positions(),
				activeAgents: envRef.current.active_agents(),
			});
		}
		setIsRunning(false);
		setIsPaused(false);
	};

	return {
		state,
		isRunning,
		isPaused,
		speed,
		actualFps,
		loadingProgress,
		loadingStatus,
		start,
		pause,
		reset,
		setSpeed,
	};
}
