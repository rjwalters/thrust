import type { UseCartPoleResult } from "./useCartPole";

interface CartPoleControlsProps {
	cartpole: UseCartPoleResult;
}

export default function CartPoleControls({ cartpole }: CartPoleControlsProps) {
	const { state, isRunning, isPaused, speed, start, pause, reset, setSpeed } =
		cartpole;

	if (!state) {
		return (
			<div className="flex items-center justify-center p-8">
				<div className="text-gray-500">Loading WASM...</div>
			</div>
		);
	}

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
				<div className="bg-gray-50 rounded-lg p-4 col-span-2">
					<div className="text-sm text-gray-600 mb-1">Best Score</div>
					<div className="text-2xl font-bold text-gray-900">
						{state.bestScore}
					</div>
				</div>
			</div>

			{/* State visualization */}
			<div className="space-y-3">
				<h3 className="text-sm font-medium text-gray-700">State</h3>

				<div className="space-y-2">
					<div className="flex justify-between text-sm">
						<span className="text-gray-600">Cart Position</span>
						<span className="font-mono text-gray-900">
							{state.position.toFixed(3)}
						</span>
					</div>
					<div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
						<div
							className="bg-blue-500 h-2 transition-all duration-100"
							style={{
								marginLeft: `${((state.position + 1) / 2) * 100}%`,
								width: "2%",
							}}
						/>
					</div>
				</div>

				<div className="space-y-2">
					<div className="flex justify-between text-sm">
						<span className="text-gray-600">Cart Velocity</span>
						<span className="font-mono text-gray-900">
							{state.velocity.toFixed(3)}
						</span>
					</div>
					<div className="w-full bg-gray-200 rounded-full h-2">
						<div
							className={`h-2 transition-all duration-100 ${
								state.velocity > 0 ? "bg-green-500" : "bg-red-500"
							}`}
							style={{
								width: `${Math.min(Math.abs(state.velocity) * 20, 100)}%`,
								marginLeft: state.velocity < 0 ? "auto" : "0",
							}}
						/>
					</div>
				</div>

				<div className="space-y-2">
					<div className="flex justify-between text-sm">
						<span className="text-gray-600">Pole Angle</span>
						<span className="font-mono text-gray-900">
							{((state.angle * 180) / Math.PI).toFixed(1)}°
						</span>
					</div>
					<div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden relative">
						<div className="absolute inset-0 flex items-center justify-center">
							<div className="w-0.5 h-full bg-gray-400" />
						</div>
						<div
							className={`h-2 transition-all duration-100 ${
								Math.abs(state.angle) > 0.2 ? "bg-red-500" : "bg-emerald-500"
							}`}
							style={{
								marginLeft: `${50 + (state.angle / 0.42) * 50}%`,
								width: "2%",
							}}
						/>
					</div>
				</div>

				<div className="space-y-2">
					<div className="flex justify-between text-sm">
						<span className="text-gray-600">Angular Velocity</span>
						<span className="font-mono text-gray-900">
							{state.angularVelocity.toFixed(3)} rad/s
						</span>
					</div>
					<div className="w-full bg-gray-200 rounded-full h-2">
						<div
							className={`h-2 transition-all duration-100 ${
								state.angularVelocity > 0 ? "bg-green-500" : "bg-red-500"
							}`}
							style={{
								width: `${Math.min(Math.abs(state.angularVelocity) * 10, 100)}%`,
								marginLeft: state.angularVelocity < 0 ? "auto" : "0",
							}}
						/>
					</div>
				</div>
			</div>

			{/* Status indicator */}
			{state.done && (
				<div className="bg-red-50 border border-red-200 rounded-lg p-4">
					<div className="text-sm font-medium text-red-800">
						Episode terminated
					</div>
					<div className="text-xs text-red-600 mt-1">
						Resetting in a moment...
					</div>
				</div>
			)}
		</div>
	);
}
