import { Link } from "react-router-dom";
import SimpleBanditVisualization from "../components/SimpleBandit/SimpleBanditVisualization";
import SimpleBanditControls from "../components/SimpleBandit/SimpleBanditControls";
import { useSimpleBandit } from "../components/SimpleBandit/useSimpleBandit";
import Footer from "../components/Footer";

export default function SimpleBanditPage() {
	const bandit = useSimpleBandit();

	return (
		<div className="min-h-screen bg-gray-50">
			<div className="container mx-auto px-4 py-8">
				<Link
					to="/"
					className="text-indigo-600 hover:text-indigo-800 mb-4 inline-block"
				>
					← Back to Home
				</Link>
				<h1 className="text-4xl font-bold mb-8">SimpleBandit</h1>

				<div className="grid lg:grid-cols-[1fr_auto] gap-8">
					{/* Visualization */}
					<div className="bg-white rounded-lg shadow-lg overflow-hidden">
						{bandit.state ? (
							<div className="w-full h-[600px]">
								<SimpleBanditVisualization state={bandit.state} />
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
						<SimpleBanditControls bandit={bandit} />
					</div>
				</div>

				{/* About Section */}
				<div className="mt-8 bg-white rounded-lg shadow-lg p-6">
					<h2 className="text-2xl font-bold mb-4">About SimpleBandit</h2>
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
							<h3 className="font-semibold mb-2">Objective</h3>
							<div className="bg-gray-50 p-3 rounded text-sm">
								<p className="text-gray-700">
									The goal is to choose the action that matches the current state:
								</p>
								<ul className="mt-2 space-y-1 text-gray-600">
									<li>• State 0 → Choose Action 0</li>
									<li>• State 1 → Choose Action 1</li>
									<li>• Maximize your success rate!</li>
								</ul>
							</div>
						</div>
					</div>

					<div className="mt-4 p-4 bg-blue-50 rounded-lg">
						<p className="text-sm text-blue-900">
							<strong>Interactive Learning:</strong> This environment runs
							entirely in your browser using WebAssembly. Make choices and see
							immediate feedback!
						</p>
					</div>
				</div>

				<Footer />
			</div>
		</div>
	);
}
