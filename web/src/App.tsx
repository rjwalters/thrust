import { BrowserRouter, Route, Routes } from "react-router-dom";
import CartPolePage from "./pages/CartPolePage";
import Home from "./pages/Home";
import SnakePage from "./pages/SnakePage";

function App() {
	return (
		<BrowserRouter basename="/thrust">
			<Routes>
				<Route path="/" element={<Home />} />
				<Route path="/cartpole" element={<CartPolePage />} />
				<Route path="/snake" element={<SnakePage />} />
			</Routes>
		</BrowserRouter>
	);
}

export default App;
