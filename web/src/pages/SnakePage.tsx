import { Link } from "react-router-dom";
import SnakeControls from "../components/Snake/SnakeControls";
import SnakePixi from "../components/Snake/SnakePixi";
import { useSnake } from "../components/Snake/useSnake";
import Footer from "../components/Footer";

export default function SnakePage() {
	const snake = useSnake();

	return (
		<div className="min-h-screen bg-gray-50">
			<div className="container mx-auto px-4 py-8">
				<Link
					to="/"
					className="text-indigo-600 hover:text-indigo-800 mb-4 inline-block"
				>
					← Back to Home
				</Link>
				<h1 className="text-4xl font-bold mb-8">Snake Game</h1>

				<div className="grid lg:grid-cols-[auto_1fr] gap-8">
					{/* Visualization */}
					<div className="bg-white rounded-lg shadow-lg p-6">
						{snake.state ? (
							<SnakePixi state={snake.state} />
						) : (
							<div className="flex items-center justify-center w-[520px] h-[520px]">
								<div className="text-gray-500">Loading...</div>
							</div>
						)}
					</div>

					{/* Controls */}
					<div className="bg-white rounded-lg shadow-lg p-6">
						<h2 className="text-2xl font-bold mb-6">Controls</h2>
						<SnakeControls snake={snake} />
					</div>
				</div>

				{/* Model Architecture */}
				<div className="mt-8 bg-white rounded-lg shadow-lg p-6">
					<h2 className="text-2xl font-bold mb-4">Neural Network Architecture</h2>
					<p className="text-gray-600 mb-4">
						Each snake agent is controlled by a Convolutional Neural Network (CNN) trained with PPO
						(Proximal Policy Optimization). The network processes the game grid and outputs actions.
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
								<div className="text-gray-500 text-xs">ReLU activation after each</div>
							</div>
						</div>

						<div>
							<h3 className="font-semibold mb-2">Fully Connected Layers</h3>
							<div className="bg-gray-50 p-3 rounded text-sm font-mono space-y-2">
								<div>Flatten: 64×20×20 = 25,600</div>
								<div>FC Common: 25,600 → 256</div>
								<div className="text-gray-500 text-xs">Shared features for policy & value</div>
							</div>
						</div>

						<div>
							<h3 className="font-semibold mb-2">Output Heads</h3>
							<div className="bg-gray-50 p-3 rounded text-sm font-mono space-y-2">
								<div>Policy: 256 → 4 actions</div>
								<div className="text-xs text-gray-600">
									(Up, Down, Left, Right)
								</div>
								<div className="mt-2">Value: 256 → 1 scalar</div>
								<div className="text-xs text-gray-600">
									(State value estimate)
								</div>
							</div>
						</div>
					</div>

					<div className="mt-4 p-4 bg-blue-50 rounded-lg">
						<p className="text-sm text-blue-900">
							<strong>Total Parameters:</strong> ~6.6M weights trained through self-play
						</p>
						<p className="text-sm text-blue-800 mt-1">
							The model runs entirely in your browser using WebAssembly for real-time inference.
						</p>
					</div>
				</div>

				<Footer />
			</div>
		</div>
	);
}
