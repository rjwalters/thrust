import { useEffect, useState } from "react";
import CartPole3D from "../components/CartPole/CartPole3D";
import CartPoleControls from "../components/CartPole/CartPoleControls";
import { useCartPole } from "../components/CartPole/useCartPole";
import GamePageLayout from "../components/GamePageLayout";

interface ModelMetadata {
	total_steps: number;
	total_episodes: number;
	final_performance: number;
	training_time_secs: number;
	device: string;
	environment: string;
	algorithm: string;
	timestamp?: string;
	notes?: string;
	hyperparameters?: Record<string, string | number | boolean>;
}

interface ModelInfo {
	obs_dim: number;
	action_dim: number;
	hidden_dim: number;
	activation: string;
	metadata?: ModelMetadata;
}

export default function CartPolePage() {
	const cartpole = useCartPole();
	const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);

	// Load model metadata
	useEffect(() => {
		async function loadModelInfo() {
			try {
				const response = await fetch("/cartpole_model.json");
				const data = await response.json();
				setModelInfo(data);
			} catch (error) {
				console.error("Failed to load model info:", error);
			}
		}
		loadModelInfo();
	}, []);

	const visualization = cartpole.state ? (
		<div className="w-full h-[600px] flex items-center justify-center">
			<CartPole3D state={cartpole.state} />
		</div>
	) : (
		<div className="flex items-center justify-center w-full h-[600px]">
			<div className="text-gray-500">Loading...</div>
		</div>
	);

	const controls = <CartPoleControls cartpole={cartpole} />;

	const gameDynamics = (
		<>
			<p className="text-gray-600 mb-4">
				CartPole is a classic control problem where the agent must balance a
				pole on a cart by applying forces left or right. The episode ends if the
				pole falls over or the cart moves too far from center.
			</p>

			<div className="grid md:grid-cols-2 gap-6">
				<div>
					<h3 className="font-semibold mb-2">State Space</h3>
					<div className="bg-gray-50 p-3 rounded text-sm space-y-1">
						<div className="text-gray-700 mb-2">4 continuous observations:</div>
						<div className="flex justify-between">
							<span className="text-gray-600">Cart Position:</span>
							<span className="font-mono text-xs">[-4.8, 4.8]</span>
						</div>
						<div className="flex justify-between">
							<span className="text-gray-600">Cart Velocity:</span>
							<span className="font-mono text-xs">[-∞, ∞]</span>
						</div>
						<div className="flex justify-between">
							<span className="text-gray-600">Pole Angle:</span>
							<span className="font-mono text-xs">[-24°, 24°]</span>
						</div>
						<div className="flex justify-between">
							<span className="text-gray-600">Pole Angular Velocity:</span>
							<span className="font-mono text-xs">[-∞, ∞]</span>
						</div>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Action Space & Rewards</h3>
					<div className="bg-gray-50 p-3 rounded text-sm space-y-2">
						<div>
							<span className="text-gray-600">Actions:</span>
							<span className="font-mono text-xs ml-2">
								2 discrete (Left/Right)
							</span>
						</div>
						<div>
							<span className="text-gray-600">Reward:</span>
							<span className="font-mono text-xs ml-2">+1 per timestep</span>
						</div>
						<div className="text-xs text-gray-600 mt-2">
							The longer the pole stays balanced, the higher the total reward.
							Episodes can last up to 500 steps.
						</div>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Termination Conditions</h3>
					<div className="bg-gray-50 p-3 rounded text-sm">
						<ul className="space-y-1 text-gray-600 text-xs">
							<li>• Pole angle exceeds ±12° from vertical</li>
							<li>• Cart position exceeds ±2.4 units from center</li>
							<li>• Episode reaches 500 timesteps (success)</li>
						</ul>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Performance Goal</h3>
					<div className="bg-gray-50 p-3 rounded text-sm">
						<p className="text-gray-700 mb-2">Considered solved when:</p>
						<div className="text-xs text-gray-600">
							Average reward ≥ 475 over 100 consecutive episodes. This trained
							agent consistently achieves 500 steps per episode.
						</div>
					</div>
				</div>
			</div>

			<div className="mt-4 p-4 bg-blue-50 rounded-lg">
				<p className="text-sm text-blue-900">
					<strong>Autonomous Agent:</strong> This environment runs entirely in
					your browser using WebAssembly. The agent plays automatically using a
					trained PPO policy.
				</p>
			</div>
		</>
	);

	const neuralNetworkArchitecture = (
		<>
			<p className="text-gray-600 mb-4">
				{modelInfo?.metadata?.algorithm || "PPO (Proximal Policy Optimization)"}{" "}
				agent trained to balance the pole on the cart.
			</p>

			<div className="grid md:grid-cols-2 gap-6">
				<div>
					<h3 className="font-semibold mb-2">Input Layer</h3>
					<div className="bg-gray-50 p-3 rounded text-sm font-mono">
						<div className="text-gray-700">
							{modelInfo?.obs_dim || 4} continuous features:
						</div>
						<ul className="mt-2 space-y-1 text-xs">
							<li>• Cart position (x)</li>
							<li>• Cart velocity (ẋ)</li>
							<li>• Pole angle (θ)</li>
							<li>• Pole angular velocity (θ̇)</li>
						</ul>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Hidden Layers</h3>
					<div className="bg-gray-50 p-3 rounded text-sm font-mono space-y-2">
						<div>
							Hidden 1: {modelInfo?.obs_dim || 4} →{" "}
							{modelInfo?.hidden_dim || 64} neurons
						</div>
						<div>
							Hidden 2: {modelInfo?.hidden_dim || 64} →{" "}
							{modelInfo?.hidden_dim || 64} neurons
						</div>
						<div className="text-gray-500 text-xs">
							{modelInfo?.activation || "ReLU"} activation
						</div>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Output Heads</h3>
					<div className="bg-gray-50 p-3 rounded text-sm font-mono space-y-2">
						<div>
							Policy: {modelInfo?.hidden_dim || 64} →{" "}
							{modelInfo?.action_dim || 2} actions
						</div>
						<div className="text-xs text-gray-600">(Push Left, Push Right)</div>
						<div className="mt-2">
							Value: {modelInfo?.hidden_dim || 64} → 1 scalar
						</div>
						<div className="text-xs text-gray-600">(State value estimate)</div>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Training Details</h3>
					<div className="bg-gray-50 p-3 rounded text-sm space-y-1">
						{modelInfo?.metadata && (
							<>
								<div className="font-mono text-xs">
									Steps: {modelInfo.metadata.total_steps.toLocaleString()}
								</div>
								<div className="font-mono text-xs">
									Episodes: {modelInfo.metadata.total_episodes.toLocaleString()}
								</div>
								<div className="font-mono text-xs">
									Performance: {modelInfo.metadata.final_performance.toFixed(1)}{" "}
									steps/ep
								</div>
								<div className="font-mono text-xs">
									Training time:{" "}
									{modelInfo.metadata.training_time_secs.toFixed(1)}s
								</div>
								<div className="font-mono text-xs">
									Device: {modelInfo.metadata.device}
								</div>
							</>
						)}
					</div>
				</div>
			</div>

			{modelInfo?.metadata?.hyperparameters && (
				<div className="mt-4">
					<h3 className="font-semibold mb-2">Hyperparameters</h3>
					<div className="bg-gray-50 p-3 rounded text-xs font-mono">
						<div className="grid grid-cols-2 gap-x-4 gap-y-1">
							{Object.entries(modelInfo.metadata.hyperparameters)
								.sort(([a], [b]) => a.localeCompare(b))
								.map(([key, value]) => (
									<div key={key} className="flex justify-between">
										<span className="text-gray-600">{key}:</span>
										<span className="font-semibold ml-2">
											{typeof value === "number" && value < 1 && value > 0
												? value.toFixed(6)
												: String(value)}
										</span>
									</div>
								))}
						</div>
					</div>
				</div>
			)}

			<div className="mt-4 p-4 bg-blue-50 rounded-lg">
				<p className="text-sm text-blue-900">
					<strong>Status:</strong>{" "}
					{cartpole.modelLoaded
						? "Model loaded successfully"
						: "Loading model..."}
				</p>
				<p className="text-sm text-blue-800 mt-1">
					The model runs entirely in your browser using WebAssembly for
					real-time inference.
				</p>
				{modelInfo?.metadata?.notes && (
					<p className="text-xs text-blue-700 mt-2 italic">
						{modelInfo.metadata.notes}
					</p>
				)}
			</div>
		</>
	);

	return (
		<GamePageLayout
			title="CartPole 3D"
			visualization={visualization}
			controls={controls}
			gameDynamics={gameDynamics}
			neuralNetworkArchitecture={neuralNetworkArchitecture}
		/>
	);
}
