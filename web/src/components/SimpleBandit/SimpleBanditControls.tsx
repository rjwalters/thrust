import type { UseSimpleBanditResult } from "./useSimpleBandit";

interface SimpleBanditControlsProps {
	bandit: UseSimpleBanditResult;
}

export default function SimpleBanditControls({ bandit }: SimpleBanditControlsProps) {
	const { state, reset, takeAction } = bandit;

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

	const getActionButtonClass = (action: number): string => {
		const baseClass =
			"px-6 py-3 rounded-lg font-bold text-white transition-all duration-200 hover:scale-105 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100";
		const colorClass =
			action === 0
				? "bg-blue-600 hover:bg-blue-700"
				: "bg-green-600 hover:bg-green-700";
		return `${baseClass} ${colorClass}`;
	};

	return (
		<div className="space-y-6">
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

			{/* Action Buttons */}
			<div className="space-y-3">
				<h3 className="text-sm font-medium text-gray-700">Choose Action</h3>
				<div className="grid grid-cols-2 gap-3">
					<button
						type="button"
						onClick={() => takeAction(0)}
						className={getActionButtonClass(0)}
					>
						Action 0
					</button>
					<button
						type="button"
						onClick={() => takeAction(1)}
						className={getActionButtonClass(1)}
					>
						Action 1
					</button>
				</div>
			</div>

			{/* Reset Button */}
			<button
				type="button"
				onClick={reset}
				className="w-full px-6 py-3 bg-gray-600 hover:bg-gray-700 text-white font-medium rounded-lg transition-colors"
			>
				Reset Episode
			</button>

			{/* Instructions */}
			<div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
				<h3 className="text-sm font-semibold text-blue-900 mb-2">
					How to Play
				</h3>
				<ul className="space-y-1 text-xs text-blue-800">
					<li>• Match action to current state</li>
					<li>• State 0 → Choose Action 0</li>
					<li>• State 1 → Choose Action 1</li>
					<li>• Correct: +1 reward</li>
					<li>• Wrong: 0 reward</li>
				</ul>
			</div>
		</div>
	);
}
