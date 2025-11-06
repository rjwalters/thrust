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
	start: () => void;
	pause: () => void;
	reset: () => void;
	setSpeed: (speed: number) => void;
}

const GRID_WIDTH = 20;
const GRID_HEIGHT = 20;
const NUM_AGENTS = 4;

export function useSnake(): UseSnakeResult {
	const [state, setState] = useState<SnakeState | null>(null);
	const [isRunning, setIsRunning] = useState(false);
	const [isPaused, setIsPaused] = useState(false);
	const [speed, setSpeed] = useState(1);

	const envRef = useRef<WasmSnake | null>(null);
	const frameIdRef = useRef<number | null>(null);
	const lastFrameTimeRef = useRef<number>(0);
	const stepsRef = useRef<number>(0);

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

			if (elapsed >= targetFrameTime) {
				lastFrameTimeRef.current = currentTime;

				// Generate random actions for all agents (placeholder for trained model)
				const actions = new Int32Array(NUM_AGENTS);
				for (let i = 0; i < NUM_AGENTS; i++) {
					actions[i] = Math.floor(Math.random() * 4); // 0: up, 1: right, 2: down, 3: left
				}

				// Step environment
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
		start,
		pause,
		reset,
		setSpeed,
	};
}
