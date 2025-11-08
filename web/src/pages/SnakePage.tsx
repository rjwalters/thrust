import { useEffect, useState } from "react";
import GamePageLayout from "../components/GamePageLayout";
import SnakeControls from "../components/Snake/SnakeControls";
import SnakePixi from "../components/Snake/SnakePixi";
import { useSnake } from "../components/Snake/useSnake";

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
	grid_width: number;
	grid_height: number;
	input_channels: number;
	num_actions: number;
	metadata?: ModelMetadata;
}

export default function SnakePage() {
	const snake = useSnake();
	const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);

	// Load model metadata
	useEffect(() => {
		async function loadModelInfo() {
			try {
				const response = await fetch(`${import.meta.env.BASE_URL}snake_model.json`);
				if (!response.ok) {
					throw new Error(`HTTP error! status: ${response.status}`);
				}
				const data = await response.json();
				setModelInfo(data);
			} catch (error) {
				console.error("Failed to load model info:", error);
			}
		}
		loadModelInfo();
	}, []);

	const visualization = snake.state ? (
		<div className="w-full flex items-center justify-center">
			<SnakePixi state={snake.state} />
		</div>
	) : (
		<div className="flex items-center justify-center w-full h-[520px]">
			<div className="max-w-md w-full px-8">
				{/* Fixed height container to prevent layout shift */}
				<div className="text-center h-24 flex flex-col justify-center mb-6">
					<div className="text-lg font-semibold text-gray-700 mb-2 min-h-[28px]">
						{snake.loadingStatus}
					</div>
					<div className="text-sm text-gray-500 min-h-[20px]">
						{snake.loadingProgress > 0 && snake.loadingProgress < 100
							? "This may take a moment on slower connections..."
							: "\u00A0"}
					</div>
				</div>

				{/* Fixed height progress bar container */}
				<div className="h-12 flex items-center">
					{snake.loadingProgress > 0 ? (
						<div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden shadow-inner">
							<div
								className="bg-gradient-to-r from-purple-500 to-pink-500 h-full rounded-full transition-all duration-300 ease-out"
								style={{ width: `${snake.loadingProgress}%` }}
							/>
						</div>
					) : (
						<div className="flex justify-center w-full">
							<div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500" />
						</div>
					)}
				</div>
			</div>
		</div>
	);

	const controls = <SnakeControls snake={snake} />;

	const gameDynamics = (
		<>
			<p className="text-gray-600 mb-4">
				Multi-agent Snake is a competitive environment where multiple snake
				agents compete for food on a shared grid. Agents must learn to navigate
				efficiently, avoid collisions with walls and other snakes, and collect
				food to grow longer and survive.
			</p>

			<div className="grid md:grid-cols-2 gap-6">
				<div>
					<h3 className="font-semibold mb-2">State Space</h3>
					<div className="bg-gray-50 p-3 rounded text-sm space-y-1">
						<div className="text-gray-700 mb-2">
							5-channel 20×20 grid observation:
						</div>
						<div className="text-xs text-gray-600 space-y-1">
							<div>• Channel 0: Own snake body</div>
							<div>• Channel 1: Own snake head</div>
							<div>• Channel 2: Other snakes</div>
							<div>• Channel 3: Food locations</div>
							<div>• Channel 4: Walls/boundaries</div>
						</div>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Action Space & Rewards</h3>
					<div className="bg-gray-50 p-3 rounded text-sm space-y-2">
						<div>
							<span className="text-gray-600">Actions:</span>
							<span className="font-mono text-xs ml-2">
								4 discrete (Up, Down, Left, Right)
							</span>
						</div>
						<div>
							<span className="text-gray-600">Rewards:</span>
							<ul className="text-xs text-gray-600 mt-1 space-y-1">
								<li>• +10 for eating food</li>
								<li>• -1 for collision (death)</li>
								<li>• Small time penalty per step</li>
							</ul>
						</div>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Termination Conditions</h3>
					<div className="bg-gray-50 p-3 rounded text-sm">
						<ul className="space-y-1 text-gray-600 text-xs">
							<li>• Snake collides with wall</li>
							<li>• Snake collides with itself</li>
							<li>• Snake collides with another snake</li>
							<li>• All snakes are eliminated</li>
						</ul>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Training Approach</h3>
					<div className="bg-gray-50 p-3 rounded text-sm">
						<div className="text-xs text-gray-600">
							Agents are trained through self-play using PPO. The population
							learns emergent behaviors like strategic positioning, efficient
							pathfinding, and avoiding other snakes while competing for food.
						</div>
					</div>
				</div>
			</div>

			<div className="mt-4 p-4 bg-blue-50 rounded-lg">
				<p className="text-sm text-blue-900">
					<strong>Autonomous Multi-Agent:</strong> This environment runs
					entirely in your browser using WebAssembly. Multiple trained agents
					compete simultaneously, demonstrating learned coordination and
					competitive behaviors.
				</p>
			</div>
		</>
	);

	const neuralNetworkArchitecture = (
		<>
			<p className="text-gray-600 mb-4">
				Each snake agent is controlled by a Convolutional Neural Network (CNN)
				trained with PPO (Proximal Policy Optimization). The network processes
				the game grid and outputs actions.
			</p>

			<div className="grid md:grid-cols-2 gap-6">
				<div>
					<h3 className="font-semibold mb-2">Input Layer</h3>
					<div className="bg-gray-50 p-3 rounded text-sm font-mono">
						<div className="text-gray-700">5-channel 20×20 grid:</div>
						<ul className="mt-2 space-y-1 text-xs">
							<li>• Channel 0: Own snake body</li>
							<li>• Channel 1: Own snake head</li>
							<li>• Channel 2: Other snakes</li>
							<li>• Channel 3: Food locations</li>
							<li>• Channel 4: Walls/boundaries</li>
						</ul>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">CNN Layers</h3>
					<div className="bg-gray-50 p-3 rounded text-sm font-mono space-y-2">
						<div>Conv1: 5 → 32 channels (3×3, pad=1)</div>
						<div>Conv2: 32 → 64 channels (3×3, pad=1)</div>
						<div>Conv3: 64 → 64 channels (3×3, pad=1)</div>
						<div className="text-gray-500 text-xs">
							ReLU activation after each
						</div>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Fully Connected Layers</h3>
					<div className="bg-gray-50 p-3 rounded text-sm font-mono space-y-2">
						<div>Flatten: 64×20×20 = 25,600</div>
						<div>FC Common: 25,600 → 256</div>
						<div className="text-gray-500 text-xs">
							Shared features for policy & value
						</div>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Output Heads</h3>
					<div className="bg-gray-50 p-3 rounded text-sm font-mono space-y-2">
						<div>Policy: 256 → 4 actions</div>
						<div className="text-xs text-gray-600">(Up, Down, Left, Right)</div>
						<div className="mt-2">Value: 256 → 1 scalar</div>
						<div className="text-xs text-gray-600">(State value estimate)</div>
					</div>
				</div>
			</div>

			<div className="mt-4">
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
								avg score
							</div>
							<div className="font-mono text-xs">
								Training time:{" "}
								{modelInfo.metadata.training_time_secs.toFixed(0)}s
							</div>
							<div className="font-mono text-xs">
								Device: {modelInfo.metadata.device}
							</div>
						</>
					)}
					{!modelInfo?.metadata && (
						<>
							<div className="font-mono text-xs">Algorithm: PPO</div>
							<div className="font-mono text-xs">Training Method: Self-play</div>
							<div className="font-mono text-xs">
								Total Parameters: ~6.6M weights
							</div>
						</>
					)}
					<div className="text-xs text-gray-600 mt-2">
						Agents learn through millions of self-play episodes, developing
						strategies for food collection, collision avoidance, and competitive
						positioning.
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
					<strong>Status:</strong> Model loaded and running
				</p>
				<p className="text-sm text-blue-800 mt-1">
					The model runs entirely in your browser using WebAssembly for
					real-time inference across multiple agents.
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
			title="Snake Game"
			visualization={visualization}
			controls={controls}
			gameDynamics={gameDynamics}
			neuralNetworkArchitecture={neuralNetworkArchitecture}
		/>
	);
}
