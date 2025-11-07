import { BrowserRouter, Route, Routes } from "react-router-dom";
import CartPolePage from "./pages/CartPolePage";
import Home from "./pages/Home";
import SimpleBanditPage from "./pages/SimpleBanditPage";
import SnakePage from "./pages/SnakePage";
import BucketBrigadePage from "./pages/BucketBrigadePage";

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

export default App;
