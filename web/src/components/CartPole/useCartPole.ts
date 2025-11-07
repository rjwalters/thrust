import { useEffect, useRef, useState } from "react";
import type { WasmCartPole } from "../../lib/wasm";
import { initWasm } from "../../lib/wasm";

export interface CartPoleState {
	position: number; // Cart x position
	velocity: number; // Cart velocity
	angle: number; // Pole angle (radians)
	angularVelocity: number; // Pole angular velocity
	episode: number;
	steps: number;
	bestScore: number;
	done: boolean;
}

export interface UseCartPoleResult {
	state: CartPoleState | null;
	isRunning: boolean;
	isPaused: boolean;
	speed: number;
	modelLoaded: boolean;
	start: () => void;
	pause: () => void;
	reset: () => void;
	setSpeed: (speed: number) => void;
}

export function useCartPole(): UseCartPoleResult {
	const [state, setState] = useState<CartPoleState | null>(null);
	const [isRunning, setIsRunning] = useState(false);
	const [isPaused, setIsPaused] = useState(false);
	const [speed, setSpeed] = useState(1);
	const [modelLoaded, setModelLoaded] = useState(false);

	const envRef = useRef<WasmCartPole | null>(null);
	const frameIdRef = useRef<number | null>(null);
	const lastFrameTimeRef = useRef<number>(0);

	// Initialize WASM environment and load policy
	useEffect(() => {
		let mounted = true;

		async function init() {
			try {
				const wasm = await initWasm();
				if (mounted) {
					envRef.current = new wasm.WasmCartPole();

					// Load the trained policy
					try {
						const response = await fetch(`${import.meta.env.BASE_URL}cartpole_model.json`);
						if (!response.ok) {
							throw new Error(`HTTP error! status: ${response.status}`);
						}
						const modelJson = await response.text();
						envRef.current.load_policy_json(modelJson);
						setModelLoaded(true);
						console.log("CartPole policy loaded successfully");
					} catch (error) {
						console.warn("Failed to load CartPole policy:", error);
						setModelLoaded(false);
					}

					const initialState = envRef.current.reset();

					setState({
						position: initialState[0],
						velocity: initialState[1],
						angle: initialState[2],
						angularVelocity: initialState[3],
						episode: envRef.current.get_episode(),
						steps: envRef.current.get_steps(),
						bestScore: envRef.current.get_best_score(),
						done: false,
					});
				}
			} catch (error) {
				console.error("Failed to initialize CartPole WASM:", error);
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

		const targetFrameTime = 16.67 / speed; // Base: 60 FPS, scales with speed

		function gameLoop(currentTime: number) {
			if (!envRef.current || !isRunning || isPaused) return;

			const elapsed = currentTime - lastFrameTimeRef.current;

			if (elapsed >= targetFrameTime) {
				lastFrameTimeRef.current = currentTime;

				// Get action from the trained policy
				const action = envRef.current.get_policy_action();

				// Step environment
				const nextState = envRef.current.step(action);
				// nextState format: [pos, vel, angle, angVel, reward, terminated, truncated]
				const done = nextState[5] === 1 || nextState[6] === 1; // terminated or truncated

				setState({
					position: nextState[0],
					velocity: nextState[1],
					angle: nextState[2],
					angularVelocity: nextState[3],
					episode: envRef.current.get_episode(),
					steps: envRef.current.get_steps(),
					bestScore: envRef.current.get_best_score(),
					done,
				});

				// Auto-reset for continuous demo
				if (done) {
					setTimeout(() => {
						if (envRef.current) {
							const resetState = envRef.current.reset();
							setState({
								position: resetState[0],
								velocity: resetState[1],
								angle: resetState[2],
								angularVelocity: resetState[3],
								episode: envRef.current.get_episode(),
								steps: envRef.current.get_steps(),
								bestScore: envRef.current.get_best_score(),
								done: false,
							});
						}
					}, 500); // Brief pause before reset
				}
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
			const resetState = envRef.current.reset();
			setState({
				position: resetState[0],
				velocity: resetState[1],
				angle: resetState[2],
				angularVelocity: resetState[3],
				episode: envRef.current.get_episode(),
				steps: envRef.current.get_steps(),
				bestScore: envRef.current.get_best_score(),
				done: false,
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
		modelLoaded,
		start,
		pause,
		reset,
		setSpeed,
	};
}
