import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import Footer from "./Footer";

interface GamePageLayoutProps {
	title: string;
	visualization: ReactNode;
	controls: ReactNode;
	gameDynamics: ReactNode;
	neuralNetworkArchitecture: ReactNode;
}

export default function GamePageLayout({
	title,
	visualization,
	controls,
	gameDynamics,
	neuralNetworkArchitecture,
}: GamePageLayoutProps) {
	return (
		<div className="min-h-screen bg-gray-50">
			<div className="container mx-auto px-4 py-8">
				<Link
					to="/"
					className="text-indigo-600 hover:text-indigo-800 mb-4 inline-block"
				>
					← Back to Home
				</Link>
				<h1 className="text-4xl font-bold mb-8">{title}</h1>

				<div className="grid lg:grid-cols-[1fr_auto] gap-8">
					{/* Visualization */}
					<div className="bg-white rounded-lg shadow-lg overflow-hidden">
						{visualization}
					</div>

					{/* Controls */}
					<div className="bg-white rounded-lg shadow-lg p-6 lg:w-80">
						<h2 className="text-2xl font-bold mb-6">Controls</h2>
						{controls}
					</div>
				</div>

				{/* Game Dynamics */}
				<div className="mt-8 bg-white rounded-lg shadow-lg p-6">
					<h2 className="text-2xl font-bold mb-4">Game Dynamics</h2>
					{gameDynamics}
				</div>

				{/* Neural Network Architecture */}
				<div className="mt-8 bg-white rounded-lg shadow-lg p-6">
					<h2 className="text-2xl font-bold mb-4">
						Neural Network Architecture
					</h2>
					{neuralNetworkArchitecture}
				</div>

				<Footer />
			</div>
		</div>
	);
}
