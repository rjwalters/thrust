import GamePageLayout from "../components/GamePageLayout";
import SimpleBanditControls from "../components/SimpleBandit/SimpleBanditControls";
import SimpleBanditVisualization from "../components/SimpleBandit/SimpleBanditVisualization";
import { useSimpleBandit } from "../components/SimpleBandit/useSimpleBandit";

export default function SimpleBanditPage() {
	const bandit = useSimpleBandit();

	const visualization = bandit.state ? (
		<div className="w-full h-[600px] flex items-center justify-center">
			<SimpleBanditVisualization state={bandit.state} />
		</div>
	) : (
		<div className="flex items-center justify-center w-full h-[600px]">
			<div className="text-gray-500">Loading...</div>
		</div>
	);

	const controls = <SimpleBanditControls bandit={bandit} />;

	const gameDynamics = (
		<>
			<p className="text-gray-600 mb-4">
				SimpleBandit is a contextual bandit environment where the agent must
				learn to match actions to states. This is one of the simplest
				reinforcement learning problems and serves as a great introduction to
				the field.
			</p>

			<div className="grid md:grid-cols-2 gap-6">
				<div>
					<h3 className="font-semibold mb-2">Environment Details</h3>
					<div className="bg-gray-50 p-3 rounded text-sm space-y-1">
						<div className="flex justify-between">
							<span className="text-gray-600">States:</span>
							<span className="font-mono">2 (0 or 1)</span>
						</div>
						<div className="flex justify-between">
							<span className="text-gray-600">Actions:</span>
							<span className="font-mono">2 (0 or 1)</span>
						</div>
						<div className="flex justify-between">
							<span className="text-gray-600">Episode Length:</span>
							<span className="font-mono">100 steps</span>
						</div>
						<div className="flex justify-between">
							<span className="text-gray-600">Reward:</span>
							<span className="font-mono">+1 (correct) / 0 (wrong)</span>
						</div>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Optimal Policy</h3>
					<div className="bg-gray-50 p-3 rounded text-sm">
						<p className="text-gray-700 mb-2">
							The agent uses a perfect matching strategy:
						</p>
						<ul className="space-y-1 text-gray-600">
							<li>• State 0 → Action 0</li>
							<li>• State 1 → Action 1</li>
							<li>• Achieves 100% success rate</li>
							<li>• Maximum reward every step</li>
						</ul>
					</div>
				</div>
			</div>

			<div className="mt-4 p-4 bg-blue-50 rounded-lg">
				<p className="text-sm text-blue-900">
					<strong>Autonomous Agent:</strong> This environment runs entirely in
					your browser using WebAssembly. The agent plays automatically using
					the optimal policy, demonstrating perfect performance on this simple
					task.
				</p>
			</div>
		</>
	);

	const neuralNetworkArchitecture = (
		<>
			<p className="text-gray-600 mb-4">
				The SimpleBandit agent uses a simple Multi-Layer Perceptron (MLP)
				trained with PPO (Proximal Policy Optimization). Given the simplicity of
				the task, the network quickly learns the optimal mapping.
			</p>

			<div className="grid md:grid-cols-2 gap-6">
				<div>
					<h3 className="font-semibold mb-2">Input Layer</h3>
					<div className="bg-gray-50 p-3 rounded text-sm font-mono">
						<div className="text-gray-700">1 discrete feature:</div>
						<ul className="mt-2 space-y-1 text-xs">
							<li>• Current state (0 or 1)</li>
						</ul>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Hidden Layers</h3>
					<div className="bg-gray-50 p-3 rounded text-sm font-mono space-y-2">
						<div>Hidden 1: 1 → 64 neurons</div>
						<div>Hidden 2: 64 → 64 neurons</div>
						<div className="text-gray-500 text-xs">ReLU activation</div>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Output Heads</h3>
					<div className="bg-gray-50 p-3 rounded text-sm font-mono space-y-2">
						<div>Policy: 64 → 2 actions</div>
						<div className="text-xs text-gray-600">(Action 0, Action 1)</div>
						<div className="mt-2">Value: 64 → 1 scalar</div>
						<div className="text-xs text-gray-600">(State value estimate)</div>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Training Details</h3>
					<div className="bg-gray-50 p-3 rounded text-sm space-y-1">
						<div className="font-mono text-xs">Algorithm: PPO</div>
						<div className="font-mono text-xs">Convergence: ~1,000 steps</div>
						<div className="font-mono text-xs">
							Final Performance: 100% success
						</div>
						<div className="text-xs text-gray-600 mt-2">
							This is a trivial problem that any working RL implementation
							should solve perfectly.
						</div>
					</div>
				</div>
			</div>

			<div className="mt-4 p-4 bg-blue-50 rounded-lg">
				<p className="text-sm text-blue-900">
					<strong>Status:</strong>{" "}
					{bandit.state ? "Model loaded successfully" : "Loading model..."}
				</p>
				<p className="text-sm text-blue-800 mt-1">
					The model runs entirely in your browser using WebAssembly for
					real-time inference.
				</p>
			</div>
		</>
	);

	return (
		<GamePageLayout
			title="SimpleBandit"
			visualization={visualization}
			controls={controls}
			gameDynamics={gameDynamics}
			neuralNetworkArchitecture={neuralNetworkArchitecture}
		/>
	);
}
