import type { UseSimpleBanditResult } from "./useSimpleBandit";

interface SimpleBanditControlsProps {
	bandit: UseSimpleBanditResult;
}

export default function SimpleBanditControls({ bandit }: SimpleBanditControlsProps) {
	const { state, isRunning, isPaused, speed, start, pause, reset, setSpeed } = bandit;

	if (!state) {
		return (
			<div className="flex items-center justify-center p-8">
				<div className="text-gray-500">Loading WASM...</div>
			</div>
		);
	}

	const getStateColor = (s: number): string => {
		return s === 0 ? "bg-blue-500" : "bg-green-500";
	};

	return (
		<div className="space-y-6">
			{/* Control buttons */}
			<div className="flex gap-3">
				{!isRunning ? (
					<button
						type="button"
						onClick={start}
						className="px-6 py-2 bg-emerald-500 hover:bg-emerald-600 text-white font-medium rounded-lg transition-colors"
					>
						Start
					</button>
				) : (
					<button
						type="button"
						onClick={pause}
						className="px-6 py-2 bg-amber-500 hover:bg-amber-600 text-white font-medium rounded-lg transition-colors"
					>
						{isPaused ? "Resume" : "Pause"}
					</button>
				)}

				<button
					type="button"
					onClick={reset}
					className="px-6 py-2 bg-gray-600 hover:bg-gray-700 text-white font-medium rounded-lg transition-colors"
				>
					Reset
				</button>
			</div>

			{/* Speed control */}
			<div className="space-y-2">
				<div className="flex justify-between items-center">
					<label htmlFor="speed" className="text-sm font-medium text-gray-700">
						Speed: {speed}x
					</label>
					<span className="text-xs text-gray-500">
						{Math.round(60 * speed)} FPS
					</span>
				</div>
				<input
					id="speed"
					type="range"
					min="0.5"
					max="5"
					step="0.5"
					value={speed}
					onChange={(e) => setSpeed(Number(e.target.value))}
					className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
				/>
			</div>

			{/* Stats */}
			<div className="grid grid-cols-2 gap-4">
				<div className="bg-gray-50 rounded-lg p-4">
					<div className="text-sm text-gray-600 mb-1">Episode</div>
					<div className="text-2xl font-bold text-gray-900">
						{state.episode}
					</div>
				</div>
				<div className="bg-gray-50 rounded-lg p-4">
					<div className="text-sm text-gray-600 mb-1">Steps</div>
					<div className="text-2xl font-bold text-gray-900">{state.steps}</div>
				</div>
				<div className="bg-gray-50 rounded-lg p-4">
					<div className="text-sm text-gray-600 mb-1">Success Rate</div>
					<div className="text-2xl font-bold text-gray-900">
						{state.successRate.toFixed(1)}%
					</div>
				</div>
				<div className="bg-gray-50 rounded-lg p-4">
					<div className="text-sm text-gray-600 mb-1">Total Reward</div>
					<div className="text-2xl font-bold text-gray-900">
						{state.totalReward.toFixed(0)}
					</div>
				</div>
			</div>

			{/* Current State Display */}
			<div className="bg-gray-50 rounded-lg p-6">
				<div className="text-center mb-4">
					<div className="text-sm text-gray-600 mb-3">Current State</div>
					<div
						className={`inline-flex items-center justify-center w-24 h-24 rounded-full ${getStateColor(
							state.state
						)} text-5xl font-bold text-white shadow-lg`}
					>
						{state.state}
					</div>
				</div>
				{state.lastReward !== null && (
					<div
						className={`text-center text-xl font-bold ${
							state.lastReward > 0 ? "text-green-600" : "text-red-600"
						}`}
					>
						{state.lastReward > 0 ? "✓ Correct!" : "✗ Wrong"}
					</div>
				)}
			</div>

			{/* Policy Information */}
			<div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
				<h3 className="text-sm font-semibold text-blue-900 mb-2">
					Optimal Policy
				</h3>
				<p className="text-xs text-blue-800 mb-2">
					This agent uses the optimal policy for SimpleBandit:
				</p>
				<ul className="space-y-1 text-xs text-blue-800">
					<li>• State 0 → Action 0 (100% success)</li>
					<li>• State 1 → Action 1 (100% success)</li>
					<li>• Perfect matching strategy</li>
				</ul>
			</div>
		</div>
	);
}
