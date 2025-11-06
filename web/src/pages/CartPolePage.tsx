import { Link } from "react-router-dom";
import { useEffect, useState } from "react";
import CartPole3D from "../components/CartPole/CartPole3D";
import CartPoleControls from "../components/CartPole/CartPoleControls";
import { useCartPole } from "../components/CartPole/useCartPole";
import Footer from "../components/Footer";

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

	return (
		<div className="min-h-screen bg-gray-50">
			<div className="container mx-auto px-4 py-8">
				<Link
					to="/"
					className="text-indigo-600 hover:text-indigo-800 mb-4 inline-block"
				>
					← Back to Home
				</Link>
				<h1 className="text-4xl font-bold mb-8">CartPole 3D</h1>

				<div className="grid lg:grid-cols-[1fr_auto] gap-8">
					{/* Visualization */}
					<div className="bg-white rounded-lg shadow-lg overflow-hidden">
						{cartpole.state ? (
							<div className="w-full h-[600px]">
								<CartPole3D state={cartpole.state} />
							</div>
						) : (
							<div className="flex items-center justify-center w-full h-[600px]">
								<div className="text-gray-500">Loading...</div>
							</div>
						)}
					</div>

					{/* Controls */}
					<div className="bg-white rounded-lg shadow-lg p-6 lg:w-80">
						<h2 className="text-2xl font-bold mb-6">Controls</h2>
						<CartPoleControls cartpole={cartpole} />
					</div>
				</div>

				{/* Model Architecture */}
				<div className="mt-8 bg-white rounded-lg shadow-lg p-6">
					<h2 className="text-2xl font-bold mb-4">Neural Network Architecture</h2>
					<p className="text-gray-600 mb-4">
						{modelInfo?.metadata?.algorithm || "PPO (Proximal Policy Optimization)"} agent trained to balance the pole on the cart.
					</p>

					<div className="grid md:grid-cols-2 gap-6">
						<div>
							<h3 className="font-semibold mb-2">Input Layer</h3>
							<div className="bg-gray-50 p-3 rounded text-sm font-mono">
								<div className="text-gray-700">{modelInfo?.obs_dim || 4} continuous features:</div>
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
								<div>Hidden 1: {modelInfo?.obs_dim || 4} → {modelInfo?.hidden_dim || 64} neurons</div>
								<div>Hidden 2: {modelInfo?.hidden_dim || 64} → {modelInfo?.hidden_dim || 64} neurons</div>
								<div className="text-gray-500 text-xs">{modelInfo?.activation || "ReLU"} activation</div>
							</div>
						</div>

						<div>
							<h3 className="font-semibold mb-2">Output Heads</h3>
							<div className="bg-gray-50 p-3 rounded text-sm font-mono space-y-2">
								<div>Policy: {modelInfo?.hidden_dim || 64} → {modelInfo?.action_dim || 2} actions</div>
								<div className="text-xs text-gray-600">
									(Push Left, Push Right)
								</div>
								<div className="mt-2">Value: {modelInfo?.hidden_dim || 64} → 1 scalar</div>
								<div className="text-xs text-gray-600">
									(State value estimate)
								</div>
							</div>
						</div>

						<div>
							<h3 className="font-semibold mb-2">Training Details</h3>
							<div className="bg-gray-50 p-3 rounded text-sm space-y-1">
								{modelInfo?.metadata && (
									<>
										<div className="font-mono text-xs">Steps: {modelInfo.metadata.total_steps.toLocaleString()}</div>
										<div className="font-mono text-xs">Episodes: {modelInfo.metadata.total_episodes.toLocaleString()}</div>
										<div className="font-mono text-xs">Performance: {modelInfo.metadata.final_performance.toFixed(1)} steps/ep</div>
										<div className="font-mono text-xs">Training time: {modelInfo.metadata.training_time_secs.toFixed(1)}s</div>
										<div className="font-mono text-xs">Device: {modelInfo.metadata.device}</div>
									</>
								)}
							</div>
						</div>
					</div>

					<div className="mt-4 p-4 bg-blue-50 rounded-lg">
						<p className="text-sm text-blue-900">
							<strong>Status:</strong> {cartpole.modelLoaded ? "Model loaded successfully" : "Loading model..."}
						</p>
						<p className="text-sm text-blue-800 mt-1">
							The model runs entirely in your browser using WebAssembly for real-time inference.
						</p>
						{modelInfo?.metadata?.notes && (
							<p className="text-xs text-blue-700 mt-2 italic">
								{modelInfo.metadata.notes}
							</p>
						)}
					</div>
				</div>

				<Footer />
			</div>
		</div>
	);
}
