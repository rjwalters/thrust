import { Link } from "react-router-dom";
import CartPole3D from "../components/CartPole/CartPole3D";
import CartPoleControls from "../components/CartPole/CartPoleControls";
import { useCartPole } from "../components/CartPole/useCartPole";

export default function CartPolePage() {
	const cartpole = useCartPole();

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
			</div>
		</div>
	);
}
