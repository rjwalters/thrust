import { BrowserRouter, Route, Routes } from "react-router-dom";
import CartPolePage from "./pages/CartPolePage";
import Home from "./pages/Home";
import PongPage from "./pages/PongPage";
import SimpleBanditPage from "./pages/SimpleBanditPage";
import SnakePage from "./pages/SnakePage";

function App() {
	return (
		<BrowserRouter basename="/thrust">
			<Routes>
				<Route path="/" element={<Home />} />
				<Route path="/cartpole" element={<CartPolePage />} />
				<Route path="/snake" element={<SnakePage />} />
				<Route path="/pong" element={<PongPage />} />
				<Route path="/bandit" element={<SimpleBanditPage />} />
			</Routes>
		</BrowserRouter>
	);
}

export default App;
