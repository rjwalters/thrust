import { useState, useEffect, useRef, useCallback } from "react";
import type { WasmSimpleBandit } from "../../lib/wasm";
import { initWasm } from "../../lib/wasm";

export interface SimpleBanditState {
	state: number;
	episode: number;
	steps: number;
	successRate: number;
	totalReward: number;
	lastReward: number | null;
}

export interface UseSimpleBanditResult {
	state: SimpleBanditState | null;
	isRunning: boolean;
	isPaused: boolean;
	speed: number;
	modelLoaded: boolean;
	start: () => void;
	pause: () => void;
	reset: () => void;
	setSpeed: (speed: number) => void;
	takeAction: (action: number) => void;
}

export function useSimpleBandit(): UseSimpleBanditResult {
	const [env, setEnv] = useState<WasmSimpleBandit | null>(null);
	const [state, setState] = useState<SimpleBanditState | null>(null);
	const [isRunning, setIsRunning] = useState(false);
	const [isPaused, setIsPaused] = useState(false);
	const [speed, setSpeed] = useState(1);
	const [modelLoaded, setModelLoaded] = useState(false);
	const animationRef = useRef<number | undefined>(undefined);
	const lastStepTimeRef = useRef<number>(0);

	// Initialize WASM and environment
	useEffect(() => {
		const loadWasm = async () => {
			try {
				const wasm = await initWasm();
				const newEnv = new wasm.WasmSimpleBandit();
				setEnv(newEnv);
				setModelLoaded(true);

				const initialStateArray = newEnv.get_state();
				setState({
					state: initialStateArray[0],
					episode: newEnv.get_episode(),
					steps: newEnv.get_steps(),
					successRate: newEnv.get_success_rate(),
					totalReward: newEnv.get_total_reward(),
					lastReward: null,
				});
			} catch (error) {
				console.error("Failed to load WASM:", error);
			}
		};

		loadWasm();

		return () => {
			if (animationRef.current) {
				cancelAnimationFrame(animationRef.current);
			}
		};
	}, []);

	const updateState = useCallback((env: WasmSimpleBandit, lastReward: number | null = null) => {
		const stateArray = env.get_state();
		setState({
			state: stateArray[0],
			episode: env.get_episode(),
			steps: env.get_steps(),
			successRate: env.get_success_rate(),
			totalReward: env.get_total_reward(),
			lastReward,
		});
	}, []);

	const reset = useCallback(() => {
		if (!env) return;
		env.reset();
		updateState(env, null);
	}, [env, updateState]);

	const takeAction = useCallback((action: number) => {
		if (!env) return;

		const result = env.step(action);
		const [, reward, terminated] = result;
		updateState(env, reward);

		if (terminated) {
			setTimeout(() => {
				if (env) {
					env.reset();
					updateState(env, null);
				}
			}, 1500);
		}
	}, [env, updateState]);

	const start = useCallback(() => {
		setIsRunning(true);
		setIsPaused(false);
	}, []);

	const pause = useCallback(() => {
		setIsPaused((prev) => !prev);
	}, []);

	// Automatic gameplay loop - optimal policy just matches action to state
	useEffect(() => {
		if (!env || !isRunning || isPaused || !state) return;

		const step = (currentTime: number) => {
			const deltaTime = currentTime - lastStepTimeRef.current;
			const stepInterval = 1000 / (60 * speed); // 60 FPS base, scaled by speed

			if (deltaTime >= stepInterval) {
				lastStepTimeRef.current = currentTime;

				// Optimal policy: action = state (perfect matching)
				const optimalAction = state.state;
				const result = env.step(optimalAction);
				const [, reward, terminated] = result;
				updateState(env, reward);

				if (terminated) {
					setTimeout(() => {
						if (env) {
							env.reset();
							updateState(env, null);
						}
					}, 1000);
				}
			}

			animationRef.current = requestAnimationFrame(step);
		};

		lastStepTimeRef.current = performance.now();
		animationRef.current = requestAnimationFrame(step);

		return () => {
			if (animationRef.current) {
				cancelAnimationFrame(animationRef.current);
			}
		};
	}, [env, isRunning, isPaused, speed, state, updateState]);

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
		takeAction,
	};
}
