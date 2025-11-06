import type { UseSnakeResult } from "./useSnake";

interface SnakeControlsProps {
	snake: UseSnakeResult;
}

const AGENT_COLORS = [
	"#10b981", // emerald
	"#3b82f6", // blue
	"#f59e0b", // amber
	"#ec4899", // pink
];

export default function SnakeControls({ snake }: SnakeControlsProps) {
	const { state, isRunning, isPaused, speed, start, pause, reset, setSpeed } =
		snake;

	if (!state) {
		return (
			<div className="flex items-center justify-center p-8">
				<div className="text-gray-500">Loading WASM...</div>
			</div>
		);
	}

	const aliveCount = Array.from(state.activeAgents).filter(
		(active) => active === 1,
	).length;

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
						{Math.round(10 * speed)} FPS
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
			</div>

			{/* Agent status */}
			<div className="space-y-2">
				<div className="text-sm font-medium text-gray-700">
					Agents Alive: {aliveCount} / {state.numAgents}
				</div>
				<div className="grid grid-cols-2 gap-2">
					{Array.from({ length: state.numAgents }, (_, i) => i).map(
						(agentId) => {
							const isAlive = state.activeAgents[agentId] === 1;
							return (
								<div
									key={`agent-${agentId}`}
									className="flex items-center gap-2 p-2 bg-gray-50 rounded"
								>
									<div
										className="w-4 h-4 rounded"
										style={{
											backgroundColor: AGENT_COLORS[agentId],
											opacity: isAlive ? 1 : 0.3,
										}}
									/>
									<span
										className={`text-sm ${isAlive ? "text-gray-900" : "text-gray-400"}`}
									>
										Agent {agentId + 1} {isAlive ? "🟢" : "💀"}
									</span>
								</div>
							);
						},
					)}
				</div>
			</div>
		</div>
	);
}
