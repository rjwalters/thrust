import { useEffect, useState } from "react";
import PongCanvas from "../components/Pong/PongCanvas";
import PongControls from "../components/Pong/PongControls";
import { fetchPongModelJson } from "../components/Pong/loadPongModel";
import { usePong } from "../components/Pong/usePong";
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

export default function PongPage() {
	const pong = usePong();
	const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);

	useEffect(() => {
		async function loadModelInfo() {
			try {
				const { json } = await fetchPongModelJson(import.meta.env.BASE_URL);
				setModelInfo(JSON.parse(json));
			} catch (e) {
				console.warn("Failed to load Pong model info:", e);
			}
		}
		loadModelInfo();
	}, []);

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

	const opponentDescription =
		pong.modelSource === "self-play"
			? "a copy of itself (red, right)"
			: "the rule-based opponent (red, right)";

	const gameDynamics = (
		<>
			<p className="text-gray-600 mb-4">
				Classic Pong: the agent (blue, left) must return the ball past{" "}
				{opponentDescription}. First to 7 points wins, or the episode ends at
				2000 steps.
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

	const hidden = modelInfo?.hidden_dim ?? 128;
	const obsDim = modelInfo?.obs_dim ?? 6;
	const actDim = modelInfo?.action_dim ?? 3;
	const activation = modelInfo?.activation ?? "Tanh";

	const neuralNetworkArchitecture = (
		<>
			<p className="text-gray-600 mb-4">
				{modelInfo?.metadata?.algorithm ?? "PPO (Proximal Policy Optimization)"}{" "}
				agent plays the left paddle.
			</p>

			<div className="grid md:grid-cols-2 gap-6">
				<div>
					<h3 className="font-semibold mb-2">Input Layer</h3>
					<div className="bg-gray-50 p-3 rounded text-sm font-mono">
						<div className="text-gray-700">{obsDim} continuous features</div>
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
						<div>Hidden 1: {obsDim} → {hidden} neurons</div>
						<div>Hidden 2: {hidden} → {hidden} neurons</div>
						<div className="text-gray-500 text-xs">{activation} activation</div>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Output Heads</h3>
					<div className="bg-gray-50 p-3 rounded text-sm font-mono space-y-2">
						<div>Policy: {hidden} → {actDim} actions</div>
						<div className="text-xs text-gray-600">(Up, Stay, Down)</div>
						<div className="mt-2">Value: {hidden} → 1 scalar</div>
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
									Avg episode: {modelInfo.metadata.final_performance.toFixed(1)} steps
								</div>
								<div className="font-mono text-xs">
									Training time: {modelInfo.metadata.training_time_secs.toFixed(1)}s
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
					{pong.modelLoaded
						? pong.modelSource === "self-play"
							? "PPO policy active (self-play trained)"
							: "PPO policy active (rule-based opponent)"
						: "No model file — using heuristic agent"}
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
			title="Pong"
			visualization={visualization}
			controls={controls}
			gameDynamics={gameDynamics}
			neuralNetworkArchitecture={neuralNetworkArchitecture}
		/>
	);
}
