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

				<Footer />
			</div>
		</div>
	);
}
