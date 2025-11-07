import type { SimpleBanditState } from "./useSimpleBandit";

interface SimpleBanditVisualizationProps {
	state: SimpleBanditState;
}

export default function SimpleBanditVisualization({
	state,
}: SimpleBanditVisualizationProps) {
	const getStateColor = (s: number): string => {
		return s === 0
			? "from-blue-500 to-blue-600"
			: "from-green-500 to-green-600";
	};

	const getStateBorderColor = (s: number): string => {
		return s === 0 ? "border-blue-400" : "border-green-400";
	};

	return (
		<div className="flex flex-col items-center justify-center h-full bg-gradient-to-br from-gray-50 to-gray-100 p-12">
			<div className="mb-8">
				<h2 className="text-2xl font-bold text-gray-700 text-center mb-2">
					Contextual Bandit
				</h2>
				<p className="text-gray-500 text-center text-sm">
					Match the action to the current state
				</p>
			</div>

			{/* State Display */}
			<div className="relative">
				<div
					className={`w-48 h-48 rounded-full bg-gradient-to-br ${getStateColor(
						state.state,
					)} border-8 ${getStateBorderColor(
						state.state,
					)} shadow-2xl flex items-center justify-center transform transition-all duration-500 ${
						state.lastReward !== null && state.lastReward > 0 ? "scale-110" : ""
					}`}
				>
					<div className="text-8xl font-bold text-white drop-shadow-lg">
						{state.state}
					</div>
				</div>

				{/* Reward Feedback */}
				{state.lastReward !== null && (
					<div
						className={`absolute -bottom-12 left-1/2 transform -translate-x-1/2 text-3xl font-bold animate-bounce ${
							state.lastReward > 0 ? "text-green-600" : "text-red-600"
						}`}
					>
						{state.lastReward > 0 ? "✓" : "✗"}
					</div>
				)}
			</div>

			{/* Progress Bar */}
			<div className="mt-20 w-full max-w-md">
				<div className="flex justify-between text-sm text-gray-600 mb-2">
					<span>Episode Progress</span>
					<span>{state.steps} / 100 steps</span>
				</div>
				<div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
					<div
						className="bg-gradient-to-r from-indigo-500 to-purple-500 h-3 transition-all duration-300 ease-out"
						style={{ width: `${(state.steps / 100) * 100}%` }}
					/>
				</div>
			</div>

			{/* Success Rate Indicator */}
			<div className="mt-8 text-center">
				<div className="text-sm text-gray-600 mb-1">Current Success Rate</div>
				<div
					className={`text-4xl font-bold ${
						state.successRate >= 80
							? "text-green-600"
							: state.successRate >= 50
								? "text-yellow-600"
								: "text-red-600"
					}`}
				>
					{state.successRate.toFixed(1)}%
				</div>
			</div>
		</div>
	);
}
