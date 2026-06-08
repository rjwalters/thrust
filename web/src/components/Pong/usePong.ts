import { useEffect, useRef, useState } from "react";
import type { WasmPong } from "../../lib/wasm";
import { initWasm } from "../../lib/wasm";
import { fetchPongModelJson, type PongModelSource } from "./loadPongModel";

export interface PongState {
	ballX: number;
	ballY: number;
	leftY: number;
	rightY: number;
	leftScore: number;
	rightScore: number;
	episode: number;
	steps: number;
	done: boolean;
}

export interface UsePongResult {
	state: PongState | null;
	isRunning: boolean;
	isPaused: boolean;
	speed: number;
	actualFps: number;
	modelLoaded: boolean;
	modelSource: PongModelSource | null;
	start: () => void;
	pause: () => void;
	reset: () => void;
	setSpeed: (speed: number) => void;
}

// Simple heuristic action when no policy loaded: track ball with left paddle
function heuristicAction(state: PongState): number {
	const threshold = 0.03;
	if (state.ballY < state.leftY - threshold) return 0; // up
	if (state.ballY > state.leftY + threshold) return 2; // down
	return 1; // stay
}

export function usePong(): UsePongResult {
	const [state, setState] = useState<PongState | null>(null);
	const [isRunning, setIsRunning] = useState(false);
	const [isPaused, setIsPaused] = useState(false);
	const [speed, setSpeed] = useState(1);
	const [actualFps, setActualFps] = useState(0);
	const [modelLoaded, setModelLoaded] = useState(false);
	const [modelSource, setModelSource] = useState<PongModelSource | null>(null);

	const envRef = useRef<WasmPong | null>(null);
	const frameIdRef = useRef<number | null>(null);
	const lastFrameTimeRef = useRef<number>(0);
	const fpsFrameTimesRef = useRef<number[]>([]);

	useEffect(() => {
		let mounted = true;

		async function init() {
			try {
				const wasm = await initWasm();
				if (!mounted) return;

				envRef.current = new wasm.WasmPong();

				try {
					const { json, source } = await fetchPongModelJson(
						import.meta.env.BASE_URL,
					);
					envRef.current.load_policy_json(json);
					setModelLoaded(true);
					setModelSource(source);
					console.log(`Pong policy loaded (${source})`);
				} catch (e) {
					console.warn("No Pong policy found, using heuristic:", e);
				}

				const s = envRef.current.reset();
				setState({
					ballX: s[0],
					ballY: s[1],
					leftY: s[2],
					rightY: s[3],
					leftScore: s[4],
					rightScore: s[5],
					episode: envRef.current.get_episode(),
					steps: envRef.current.get_steps(),
					done: false,
				});
			} catch (e) {
				console.error("Failed to init Pong WASM:", e);
			}
		}

		init();
		return () => {
			mounted = false;
			if (frameIdRef.current !== null) cancelAnimationFrame(frameIdRef.current);
		};
	}, []);

	useEffect(() => {
		if (!isRunning || isPaused || !envRef.current) return;

		const targetFrameTime = 16.67 / speed;

		function gameLoop(currentTime: number) {
			if (!envRef.current || !isRunning || isPaused) return;

			fpsFrameTimesRef.current.push(currentTime);
			if (fpsFrameTimesRef.current.length > 30) fpsFrameTimesRef.current.shift();
			if (fpsFrameTimesRef.current.length >= 2) {
				const span = currentTime - fpsFrameTimesRef.current[0];
				setActualFps(
					Math.round(((fpsFrameTimesRef.current.length - 1) / span) * 1000),
				);
			}

			const elapsed = currentTime - lastFrameTimeRef.current;
			if (elapsed >= targetFrameTime) {
				lastFrameTimeRef.current = currentTime;

				// Get action
				let action = envRef.current.get_policy_action();
				if (action === -1) {
					// No policy — use heuristic based on current state
					const cur = envRef.current.get_state();
					const curState: PongState = {
						ballX: cur[0], ballY: cur[1],
						leftY: cur[2], rightY: cur[3],
						leftScore: cur[4], rightScore: cur[5],
						episode: 0, steps: 0, done: false,
					};
					action = heuristicAction(curState);
				}

				const result = envRef.current.step(action);
				const done = result[7] === 1.0 || result[8] === 1.0;

				const next: PongState = {
					ballX: result[0],
					ballY: result[1],
					leftY: result[2],
					rightY: result[3],
					leftScore: result[4],
					rightScore: result[5],
					episode: envRef.current.get_episode(),
					steps: envRef.current.get_steps(),
					done,
				};
				setState(next);

				if (done) {
					setTimeout(() => {
						if (envRef.current) {
							const s = envRef.current.reset();
							setState({
								ballX: s[0], ballY: s[1],
								leftY: s[2], rightY: s[3],
								leftScore: s[4], rightScore: s[5],
								episode: envRef.current.get_episode(),
								steps: envRef.current.get_steps(),
								done: false,
							});
						}
					}, 800);
				}
			}

			frameIdRef.current = requestAnimationFrame(gameLoop);
		}

		lastFrameTimeRef.current = performance.now();
		frameIdRef.current = requestAnimationFrame(gameLoop);

		return () => {
			if (frameIdRef.current !== null) cancelAnimationFrame(frameIdRef.current);
		};
	}, [isRunning, isPaused, speed]);

	const start = () => {
		if (!isRunning && envRef.current) {
			setIsRunning(true);
			setIsPaused(false);
		}
	};

	const pause = () => setIsPaused((p) => !p);

	const reset = () => {
		if (envRef.current) {
			const s = envRef.current.reset();
			setState({
				ballX: s[0], ballY: s[1],
				leftY: s[2], rightY: s[3],
				leftScore: s[4], rightScore: s[5],
				episode: envRef.current.get_episode(),
				steps: envRef.current.get_steps(),
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
		actualFps,
		modelLoaded,
		modelSource,
		start,
		pause,
		reset,
		setSpeed,
	};
}
