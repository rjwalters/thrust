import PongCanvas from "../components/Pong/PongCanvas";
import PongControls from "../components/Pong/PongControls";
import { usePong } from "../components/Pong/usePong";
import GamePageLayout from "../components/GamePageLayout";

export default function PongPage() {
	const pong = usePong();

	const visualization = pong.state ? (
		<div className="w-full h-[400px] flex items-center justify-center bg-gray-900">
			<PongCanvas state={pong.state} />
		</div>
	) : (
		<div className="flex items-center justify-center w-full h-[400px] bg-gray-900">
			<div className="text-gray-400">Loading...</div>
		</div>
	);

	const controls = <PongControls pong={pong} />;

	const gameDynamics = (
		<>
			<p className="text-gray-600 mb-4">
				Classic Pong: the agent (blue, left) must return the ball past the
				rule-based opponent (red, right). First to 7 points wins, or the episode
				ends at 2000 steps.
			</p>

			<div className="grid md:grid-cols-2 gap-6">
				<div>
					<h3 className="font-semibold mb-2">Observation Space</h3>
					<div className="bg-gray-50 p-3 rounded text-sm space-y-1">
						<div className="text-gray-700 mb-2">6 continuous features (all in [-1, 1]):</div>
						<div className="flex justify-between">
							<span className="text-gray-600">Ball X</span>
							<span className="font-mono text-xs">position</span>
						</div>
						<div className="flex justify-between">
							<span className="text-gray-600">Ball Y</span>
							<span className="font-mono text-xs">position</span>
						</div>
						<div className="flex justify-between">
							<span className="text-gray-600">Ball dX</span>
							<span className="font-mono text-xs">velocity</span>
						</div>
						<div className="flex justify-between">
							<span className="text-gray-600">Ball dY</span>
							<span className="font-mono text-xs">velocity</span>
						</div>
						<div className="flex justify-between">
							<span className="text-gray-600">Agent paddle Y</span>
							<span className="font-mono text-xs">position</span>
						</div>
						<div className="flex justify-between">
							<span className="text-gray-600">Opponent paddle Y</span>
							<span className="font-mono text-xs">position</span>
						</div>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Action Space & Rewards</h3>
					<div className="bg-gray-50 p-3 rounded text-sm space-y-2">
						<div>
							<span className="text-gray-600">Actions:</span>
							<span className="font-mono text-xs ml-2">3 discrete (Up / Stay / Down)</span>
						</div>
						<div>
							<span className="text-gray-600">Score point:</span>
							<span className="font-mono text-xs ml-2">+1.0</span>
						</div>
						<div>
							<span className="text-gray-600">Concede point:</span>
							<span className="font-mono text-xs ml-2">-1.0</span>
						</div>
						<div>
							<span className="text-gray-600">Return ball:</span>
							<span className="font-mono text-xs ml-2">+0.1</span>
						</div>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Opponent AI</h3>
					<div className="bg-gray-50 p-3 rounded text-sm">
						<p className="text-gray-600 text-xs">
							The right paddle tracks the ball at 60% of the agent's maximum
							speed, making it beatable through spin and placement. The agent
							learns to exploit the speed gap by angling shots to corners.
						</p>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Ball Physics</h3>
					<div className="bg-gray-50 p-3 rounded text-sm text-xs text-gray-600 space-y-1">
						<div>• Constant speed in X direction</div>
						<div>• Paddle spin adjusts Y trajectory on hit</div>
						<div>• Top/bottom walls reflect ball</div>
						<div>• Ball resets to center after each score</div>
					</div>
				</div>
			</div>

			<div className="mt-4 p-4 bg-blue-50 rounded-lg">
				<p className="text-sm text-blue-900">
					<strong>Autonomous Agent:</strong> Runs entirely in your browser via
					WebAssembly. Uses a trained PPO policy; falls back to a simple
					ball-tracking heuristic if no model is loaded.
				</p>
			</div>
		</>
	);

	const neuralNetworkArchitecture = (
		<>
			<p className="text-gray-600 mb-4">
				PPO (Proximal Policy Optimization) agent trained against a rule-based
				opponent.
			</p>

			<div className="grid md:grid-cols-2 gap-6">
				<div>
					<h3 className="font-semibold mb-2">Input Layer</h3>
					<div className="bg-gray-50 p-3 rounded text-sm font-mono">
						<div className="text-gray-700">6 continuous features</div>
						<ul className="mt-2 space-y-1 text-xs text-gray-600">
							<li>• Ball position (x, y)</li>
							<li>• Ball velocity (dx, dy)</li>
							<li>• Agent paddle Y</li>
							<li>• Opponent paddle Y</li>
						</ul>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Hidden Layers</h3>
					<div className="bg-gray-50 p-3 rounded text-sm font-mono space-y-2">
						<div>Hidden 1: 6 → 128 neurons</div>
						<div>Hidden 2: 128 → 128 neurons</div>
						<div className="text-gray-500 text-xs">ReLU activation</div>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Output Heads</h3>
					<div className="bg-gray-50 p-3 rounded text-sm font-mono space-y-2">
						<div>Policy: 128 → 3 actions</div>
						<div className="text-xs text-gray-600">(Up, Stay, Down)</div>
						<div className="mt-2">Value: 128 → 1 scalar</div>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Training Setup</h3>
					<div className="bg-gray-50 p-3 rounded text-sm text-xs text-gray-600 space-y-1">
						<div>• 32 parallel environments</div>
						<div>• 128 steps per rollout</div>
						<div>• γ = 0.99, λ = 0.95</div>
						<div>• Opponent speed 60% of agent</div>
					</div>
				</div>
			</div>

			<div className="mt-4 p-4 bg-blue-50 rounded-lg">
				<p className="text-sm text-blue-900">
					<strong>Status:</strong>{" "}
					{pong.modelLoaded
						? "Model loaded — PPO policy active"
						: "No model file — using heuristic agent (train and deploy pong_model.json to activate)"}
				</p>
			</div>
		</>
	);

	return (
		<GamePageLayout
			title="Pong"
			visualization={visualization}
			controls={controls}
			gameDynamics={gameDynamics}
			neuralNetworkArchitecture={neuralNetworkArchitecture}
		/>
	);
}
