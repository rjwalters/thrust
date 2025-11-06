import { Link } from "react-router-dom";
import Footer from "../components/Footer";

export default function Home() {
	return (
		<div className="min-h-screen bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-600">
			<div className="container mx-auto px-4 py-16">
				<div className="text-center mb-16">
					<h1 className="text-6xl font-bold text-white mb-4">Thrust RL</h1>
					<p className="text-xl text-white/90 max-w-2xl mx-auto">
						High-performance reinforcement learning in Rust + CUDA. Watch AI
						agents learn in real-time, compiled to WebAssembly.
					</p>
				</div>

				<div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
					<Link
						to="/cartpole"
						className="group bg-white/10 backdrop-blur-lg rounded-2xl p-8 hover:bg-white/20 transition-all hover:scale-105 border border-white/20"
					>
						<h2 className="text-3xl font-bold text-white mb-4">CartPole 3D</h2>
						<p className="text-white/80 mb-4">
							Classic control problem visualized in 3D with Three.js. Watch the
							pole balance in real-time with realistic physics.
						</p>
						<div className="text-sm text-white/60">
							<span className="bg-white/20 px-3 py-1 rounded-full">
								Three.js
							</span>
							<span className="bg-white/20 px-3 py-1 rounded-full ml-2">
								3D Graphics
							</span>
						</div>
					</Link>

					<Link
						to="/snake"
						className="group bg-white/10 backdrop-blur-lg rounded-2xl p-8 hover:bg-white/20 transition-all hover:scale-105 border border-white/20"
					>
						<h2 className="text-3xl font-bold text-white mb-4">
							Multi-Agent Snake
						</h2>
						<p className="text-white/80 mb-4">
							4 AI agents competing in Snake with hardware-accelerated rendering
							using Pixi.js.
						</p>
						<div className="text-sm text-white/60">
							<span className="bg-white/20 px-3 py-1 rounded-full">
								Pixi.js
							</span>
							<span className="bg-white/20 px-3 py-1 rounded-full ml-2">
								Multi-Agent
							</span>
						</div>
					</Link>
				</div>

				<div className="mt-16 text-center">
					<a
						href="https://github.com/yourusername/thrust"
						className="inline-block bg-white/20 backdrop-blur-lg text-white px-8 py-3 rounded-full hover:bg-white/30 transition-all border border-white/20"
					>
						View on GitHub
					</a>
				</div>

				<Footer />
			</div>
		</div>
	);
}
