import { BrowserRouter, Route, Routes } from "react-router-dom";
import CartPolePage from "./pages/CartPolePage";
import Home from "./pages/Home";
import SnakePage from "./pages/SnakePage";
import SimpleBanditPage from "./pages/SimpleBanditPage";

function App() {
	return (
		<BrowserRouter basename="/thrust">
			<Routes>
				<Route path="/" element={<Home />} />
				<Route path="/cartpole" element={<CartPolePage />} />
				<Route path="/snake" element={<SnakePage />} />
				<Route path="/bandit" element={<SimpleBanditPage />} />
				<Route path="/bucket-brigade" element={<BucketBrigadePage />} />
			</Routes>
		</BrowserRouter>
	);
}

function BucketBrigadePage() {
	return (
		<div className="min-h-screen bg-gray-900 text-white p-8">
			<div className="max-w-4xl mx-auto text-center">
				<h1 className="text-4xl font-bold mb-4">Bucket Brigade</h1>
				<p className="text-gray-400 mb-8">
					A multi-agent cooperative environment where agents work together to fight fires.
				</p>
				<div className="bg-gray-800 rounded-lg p-6">
					<p className="mb-4">
						The Bucket Brigade environment has its own dedicated visualizer.
					</p>
					<a
						href="https://rjwalters.github.io/bucket-brigade/"
						target="_blank"
						rel="noopener noreferrer"
						className="inline-block px-8 py-4 bg-blue-600 hover:bg-blue-700 rounded-lg font-bold transition-all duration-200"
					>
						Open Bucket Brigade Visualizer
					</a>
				</div>
			</div>
		</div>
	);
}

export default App;
