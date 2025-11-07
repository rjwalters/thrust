import { useState } from "react";
import GamePageLayout from "../components/GamePageLayout";
import Town from "../components/BucketBrigade/Town";

type HouseState = 0 | 1 | 2; // SAFE = 0, BURNING = 1, RUINED = 2

export default function BucketBrigadePage() {
	// Demo state - replace with actual game logic later
	const [houses] = useState<HouseState[]>([0, 0, 1, 0, 2, 0, 1, 0, 0, 0]);
	const numAgents = 4;
	const archetypes = ["firefighter", "coordinator", "hero", "free_rider"];

	const visualization = (
		<div className="w-full h-[640px] relative">
			<Town houses={houses} numAgents={numAgents} archetypes={archetypes} />
		</div>
	);

	const controls = (
		<div className="space-y-4">
			<div className="bg-gray-50 p-4 rounded-lg">
				<h3 className="font-semibold mb-2">Demo Mode</h3>
				<p className="text-sm text-gray-600">
					This is a static visualization of the Bucket Brigade environment.
					Interactive gameplay coming soon!
				</p>
			</div>
			<div className="space-y-2">
				<div className="flex justify-between items-center p-2 bg-white rounded border">
					<span className="text-sm font-medium">Houses Safe:</span>
					<span className="text-sm">
						{houses.filter((h) => h === 0).length}/10
					</span>
				</div>
				<div className="flex justify-between items-center p-2 bg-white rounded border">
					<span className="text-sm font-medium">Houses Burning:</span>
					<span className="text-sm">
						{houses.filter((h) => h === 1).length}/10
					</span>
				</div>
				<div className="flex justify-between items-center p-2 bg-white rounded border">
					<span className="text-sm font-medium">Houses Ruined:</span>
					<span className="text-sm">
						{houses.filter((h) => h === 2).length}/10
					</span>
				</div>
			</div>
		</div>
	);

	const gameDynamics = (
		<>
			<p className="text-gray-600 mb-4">
				Bucket Brigade is a multi-agent cooperative environment where agents
				must work together to fight fires in a town of 10 houses arranged in a
				circle. Each agent owns specific houses and must decide whether to
				protect their own property or help neighbors.
			</p>

			<div className="grid md:grid-cols-2 gap-6">
				<div>
					<h3 className="font-semibold mb-2">Environment Setup</h3>
					<div className="bg-gray-50 p-3 rounded text-sm space-y-1">
						<div className="text-gray-700 mb-2">Town structure:</div>
						<div className="flex justify-between">
							<span className="text-gray-600">Houses:</span>
							<span className="font-mono text-xs">10 in circle</span>
						</div>
						<div className="flex justify-between">
							<span className="text-gray-600">Agents:</span>
							<span className="font-mono text-xs">2-10 players</span>
						</div>
						<div className="flex justify-between">
							<span className="text-gray-600">Ownership:</span>
							<span className="font-mono text-xs">Round-robin</span>
						</div>
						<div className="flex justify-between">
							<span className="text-gray-600">Fire spread:</span>
							<span className="font-mono text-xs">Adjacent houses</span>
						</div>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Actions & Rewards</h3>
					<div className="bg-gray-50 p-3 rounded text-sm space-y-2">
						<div>
							<span className="text-gray-600">Actions:</span>
							<ul className="text-xs text-gray-600 mt-1 space-y-1">
								<li>• Fight fire at any house</li>
								<li>• Do nothing (rest)</li>
								<li>• Strategic positioning</li>
							</ul>
						</div>
						<div>
							<span className="text-gray-600">Rewards:</span>
							<ul className="text-xs text-gray-600 mt-1 space-y-1">
								<li>• Bonus for houses saved</li>
								<li>• Penalty for houses lost</li>
								<li>• Team vs individual rewards</li>
							</ul>
						</div>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">House States</h3>
					<div className="bg-gray-50 p-3 rounded text-sm">
						<ul className="space-y-1 text-gray-600 text-xs">
							<li>• Safe (🏠): Not on fire, can catch fire from neighbors</li>
							<li>• Burning (🔥): On fire, needs agents to extinguish</li>
							<li>• Ruined (💀): Burned down, cannot be saved</li>
						</ul>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Agent Archetypes</h3>
					<div className="bg-gray-50 p-3 rounded text-sm">
						<ul className="space-y-1 text-gray-600 text-xs">
							<li>• <strong>Firefighter:</strong> Cooperative, helps everyone</li>
							<li>• <strong>Coordinator:</strong> Strategic team player</li>
							<li>• <strong>Hero:</strong> High-effort individual</li>
							<li>• <strong>Free-rider:</strong> Relies on others</li>
						</ul>
					</div>
				</div>
			</div>

			<div className="mt-4 p-4 bg-blue-50 rounded-lg">
				<p className="text-sm text-blue-900">
					<strong>Research Focus:</strong> This environment is designed to
					study cooperation, coordination, and social dilemmas in multi-agent
					systems. It tests whether agents learn to cooperate or act
					selfishly.
				</p>
			</div>
		</>
	);

	const neuralNetworkArchitecture = (
		<>
			<p className="text-gray-600 mb-4">
				Agents use heuristic strategies parameterized by behavioral traits.
				Future versions will include neural network policies trained with
				multi-agent reinforcement learning (MARL).
			</p>

			<div className="grid md:grid-cols-2 gap-6">
				<div>
					<h3 className="font-semibold mb-2">Current Approach</h3>
					<div className="bg-gray-50 p-3 rounded text-sm font-mono space-y-2">
						<div className="text-gray-700">Heuristic Agents:</div>
						<ul className="mt-2 space-y-1 text-xs">
							<li>• 10 behavioral parameters</li>
							<li>• Risk tolerance</li>
							<li>• Cooperation willingness</li>
							<li>• Strategic positioning</li>
							<li>• Effort allocation</li>
						</ul>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Training Methods</h3>
					<div className="bg-gray-50 p-3 rounded text-sm font-mono space-y-2">
						<div>Genetic Algorithms:</div>
						<div className="text-xs text-gray-600">
							Population-based evolution of strategies
						</div>
						<div className="mt-2">Nash Equilibrium:</div>
						<div className="text-xs text-gray-600">
							Game-theoretic optimal strategies
						</div>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Future: Neural Policies</h3>
					<div className="bg-gray-50 p-3 rounded text-sm font-mono space-y-2">
						<div>Input: Multi-agent observations</div>
						<div>CNN/GNN: Process spatial town layout</div>
						<div>Attention: Model agent interactions</div>
						<div>Output: Continuous action probabilities</div>
					</div>
				</div>

				<div>
					<h3 className="font-semibold mb-2">Research Questions</h3>
					<div className="bg-gray-50 p-3 rounded text-sm space-y-1">
						<div className="text-xs text-gray-600">
							• How do reward structures affect cooperation?
						</div>
						<div className="text-xs text-gray-600">
							• Can agents learn to communicate implicitly?
						</div>
						<div className="text-xs text-gray-600">
							• What strategies emerge in different scenarios?
						</div>
						<div className="text-xs text-gray-600">
							• How does team size affect coordination?
						</div>
					</div>
				</div>
			</div>

			<div className="mt-4 p-4 bg-blue-50 rounded-lg">
				<p className="text-sm text-blue-900">
					<strong>Status:</strong> Demo visualization with static state
				</p>
				<p className="text-sm text-blue-800 mt-1">
					The full interactive environment with trained agents is available at
					the dedicated{" "}
					<a
						href="https://rjwalters.github.io/bucket-brigade/"
						target="_blank"
						rel="noopener noreferrer"
						className="underline hover:text-blue-900"
					>
						Bucket Brigade research platform
					</a>
					.
				</p>
			</div>
		</>
	);

	return (
		<GamePageLayout
			title="Bucket Brigade"
			visualization={visualization}
			controls={controls}
			gameDynamics={gameDynamics}
			neuralNetworkArchitecture={neuralNetworkArchitecture}
		/>
	);
}
