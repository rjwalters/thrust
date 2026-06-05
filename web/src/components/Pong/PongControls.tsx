import type { UsePongResult } from "./usePong";

interface PongControlsProps {
	pong: UsePongResult;
}

export default function PongControls({ pong }: PongControlsProps) {
	const { state, isRunning, isPaused, speed, actualFps, modelLoaded, start, pause, reset, setSpeed } =
		pong;

	if (!state) {
		return (
			<div className="flex items-center justify-center p-8">
				<div className="text-gray-500">Loading WASM...</div>
			</div>
		);
	}

	return (
		<div className="space-y-6">
			{/* Controls */}
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

			{/* Speed */}
			<div className="space-y-2">
				<div className="flex justify-between items-center">
					<label htmlFor="pong-speed" className="text-sm font-medium text-gray-700">
						Speed: {speed}x
					</label>
					<div className="flex gap-2 text-xs">
						<span className="text-gray-500">Target: {Math.round(60 * speed)} FPS</span>
						{isRunning && actualFps > 0 && (
							<span className="text-emerald-600 font-semibold">
								• Actual: {actualFps} FPS
							</span>
						)}
					</div>
				</div>
				<input
					id="pong-speed"
					type="range"
					min="0.5"
					max="5"
					step="0.5"
					value={speed}
					onChange={(e) => setSpeed(Number(e.target.value))}
					className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
				/>
			</div>

			{/* Scoreboard */}
			<div className="grid grid-cols-2 gap-3">
				<div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-center">
					<div className="text-xs text-blue-600 font-medium mb-1">Agent</div>
					<div className="text-4xl font-bold text-blue-700">{Math.round(state.leftScore)}</div>
				</div>
				<div className="bg-red-50 border border-red-200 rounded-lg p-4 text-center">
					<div className="text-xs text-red-600 font-medium mb-1">Opponent</div>
					<div className="text-4xl font-bold text-red-700">{Math.round(state.rightScore)}</div>
				</div>
			</div>

			{/* Stats */}
			<div className="grid grid-cols-2 gap-3">
				<div className="bg-gray-50 rounded-lg p-3">
					<div className="text-xs text-gray-500 mb-1">Episode</div>
					<div className="text-xl font-bold text-gray-800">{state.episode}</div>
				</div>
				<div className="bg-gray-50 rounded-lg p-3">
					<div className="text-xs text-gray-500 mb-1">Steps</div>
					<div className="text-xl font-bold text-gray-800">{state.steps}</div>
				</div>
			</div>

			{/* Policy status */}
			<div
				className={`rounded-lg p-3 text-sm ${
					modelLoaded
						? "bg-emerald-50 text-emerald-800"
						: "bg-amber-50 text-amber-800"
				}`}
			>
				{modelLoaded ? "PPO policy loaded" : "No policy — using heuristic"}
			</div>

			{state.done && (
				<div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3 text-sm text-indigo-800">
					Episode ended — resetting...
				</div>
			)}
		</div>
	);
}
