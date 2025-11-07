import { useState, useEffect, useRef } from 'react';
import init, { WasmSimpleBandit } from '../../public/pkg';

const SimpleBanditPage = () => {
    const [env, setEnv] = useState<WasmSimpleBandit | null>(null);
    const [state, setState] = useState<number>(0);
    const [episode, setEpisode] = useState<number>(0);
    const [steps, setSteps] = useState<number>(0);
    const [successRate, setSuccessRate] = useState<number>(0);
    const [totalReward, setTotalReward] = useState<number>(0);
    const [lastReward, setLastReward] = useState<number | null>(null);
    const [isRunning, setIsRunning] = useState<boolean>(false);
    const animationRef = useRef<number>();

    useEffect(() => {
        const loadWasm = async () => {
            try {
                await init();
                const newEnv = new WasmSimpleBandit();
                setEnv(newEnv);
                const initialState = newEnv.get_state();
                setState(initialState[0]);
            } catch (error) {
                console.error('Failed to load WASM:', error);
            }
        };

        loadWasm();

        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, []);

    const reset = () => {
        if (!env) return;
        const newState = env.reset();
        setState(newState[0]);
        setEpisode(env.get_episode());
        setSteps(env.get_steps());
        setSuccessRate(env.get_success_rate());
        setTotalReward(env.get_total_reward());
        setLastReward(null);
    };

    const takeAction = (action: number) => {
        if (!env) return;

        const result = env.step(action);
        const [newState, reward, terminated] = result;

        setState(newState);
        setSteps(env.get_steps());
        setSuccessRate(env.get_success_rate());
        setTotalReward(env.get_total_reward());
        setLastReward(reward);

        if (terminated) {
            setTimeout(reset, 1500);
        }
    };

    const getStateColor = (s: number): string => {
        return s === 0 ? 'bg-blue-500' : 'bg-green-500';
    };

    const getActionButtonClass = (action: number): string => {
        const baseClass = 'px-8 py-4 rounded-lg font-bold text-white transition-all duration-200 hover:scale-105 active:scale-95';
        const colorClass = action === 0 ? 'bg-blue-600 hover:bg-blue-700' : 'bg-green-600 hover:bg-green-700';
        return `${baseClass} ${colorClass}`;
    };

    return (
        <div className="min-h-screen bg-gray-900 text-white p-8">
            <div className="max-w-4xl mx-auto">
                <h1 className="text-4xl font-bold mb-2 text-center">SimpleBandit</h1>
                <p className="text-gray-400 text-center mb-8">
                    A contextual bandit environment. Choose the action that matches the state!
                </p>

                {/* Stats Panel */}
                <div className="grid grid-cols-4 gap-4 mb-8">
                    <div className="bg-gray-800 rounded-lg p-4">
                        <div className="text-gray-400 text-sm">Episode</div>
                        <div className="text-2xl font-bold">{episode}</div>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4">
                        <div className="text-gray-400 text-sm">Steps</div>
                        <div className="text-2xl font-bold">{steps}</div>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4">
                        <div className="text-gray-400 text-sm">Success Rate</div>
                        <div className="text-2xl font-bold">{successRate.toFixed(1)}%</div>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4">
                        <div className="text-gray-400 text-sm">Total Reward</div>
                        <div className="text-2xl font-bold">{totalReward.toFixed(1)}</div>
                    </div>
                </div>

                {/* Current State Display */}
                <div className="bg-gray-800 rounded-lg p-8 mb-8">
                    <div className="text-center mb-4">
                        <div className="text-gray-400 text-sm mb-2">Current State</div>
                        <div className={`inline-flex items-center justify-center w-32 h-32 rounded-full ${getStateColor(state)} text-6xl font-bold`}>
                            {state}
                        </div>
                    </div>
                    {lastReward !== null && (
                        <div className={`text-center text-2xl font-bold ${lastReward > 0 ? 'text-green-400' : 'text-red-400'}`}>
                            {lastReward > 0 ? '✓ Correct!' : '✗ Wrong'}
                        </div>
                    )}
                </div>

                {/* Action Buttons */}
                <div className="grid grid-cols-2 gap-4 mb-8">
                    <button
                        onClick={() => takeAction(0)}
                        disabled={!env}
                        className={getActionButtonClass(0)}
                    >
                        Action 0
                    </button>
                    <button
                        onClick={() => takeAction(1)}
                        disabled={!env}
                        className={getActionButtonClass(1)}
                    >
                        Action 1
                    </button>
                </div>

                {/* Reset Button */}
                <div className="text-center">
                    <button
                        onClick={reset}
                        disabled={!env}
                        className="px-6 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg font-bold transition-all duration-200"
                    >
                        Reset Episode
                    </button>
                </div>

                {/* Instructions */}
                <div className="mt-8 bg-gray-800 rounded-lg p-6">
                    <h2 className="text-xl font-bold mb-3">How to Play</h2>
                    <ul className="space-y-2 text-gray-300">
                        <li>• The state is shown in the circle (0 or 1)</li>
                        <li>• Choose the action button that matches the current state</li>
                        <li>• Correct choices give +1 reward, wrong choices give 0</li>
                        <li>• Each episode lasts 100 steps</li>
                        <li>• Try to achieve 100% success rate!</li>
                    </ul>
                </div>
            </div>
        </div>
    );
};

export default SimpleBanditPage;
