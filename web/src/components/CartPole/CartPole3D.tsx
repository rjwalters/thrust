import { Environment, Grid, OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import type { CartPoleState } from "./useCartPole";

interface CartPole3DProps {
	state: CartPoleState;
}

const TRACK_LENGTH = 4.8; // CartPole track is 4.8 units wide
const CART_WIDTH = 0.5;
const CART_HEIGHT = 0.3;
const CART_DEPTH = 0.3;
const POLE_LENGTH = 1.0;
const POLE_RADIUS = 0.05;

function Track() {
	return (
		<>
			{/* Track base */}
			<mesh position={[0, -0.05, 0]} receiveShadow>
				<boxGeometry args={[TRACK_LENGTH, 0.1, 0.4]} />
				<meshStandardMaterial color="#444" />
			</mesh>

			{/* Track rails */}
			<mesh position={[0, 0.05, 0.15]} castShadow>
				<boxGeometry args={[TRACK_LENGTH, 0.05, 0.05]} />
				<meshStandardMaterial color="#888" metalness={0.6} roughness={0.4} />
			</mesh>
			<mesh position={[0, 0.05, -0.15]} castShadow>
				<boxGeometry args={[TRACK_LENGTH, 0.05, 0.05]} />
				<meshStandardMaterial color="#888" metalness={0.6} roughness={0.4} />
			</mesh>
		</>
	);
}

function Cart({ position, angle }: { position: number; angle: number }) {
	// Position is normalized to [-1, 1], scale to track length
	const xPosition = position * (TRACK_LENGTH / 2);

	return (
		<group position={[xPosition, CART_HEIGHT / 2, 0]}>
			{/* Cart body */}
			<mesh castShadow receiveShadow>
				<boxGeometry args={[CART_WIDTH, CART_HEIGHT, CART_DEPTH]} />
				<meshStandardMaterial
					color={angle > 0.2 || angle < -0.2 ? "#ef4444" : "#3b82f6"}
					metalness={0.3}
					roughness={0.7}
				/>
			</mesh>

			{/* Wheels */}
			<mesh position={[-0.15, -CART_HEIGHT / 2, 0.2]} castShadow>
				<cylinderGeometry args={[0.08, 0.08, 0.05, 16]} />
				<meshStandardMaterial color="#222" />
			</mesh>
			<mesh position={[0.15, -CART_HEIGHT / 2, 0.2]} castShadow>
				<cylinderGeometry args={[0.08, 0.08, 0.05, 16]} />
				<meshStandardMaterial color="#222" />
			</mesh>
			<mesh position={[-0.15, -CART_HEIGHT / 2, -0.2]} castShadow>
				<cylinderGeometry args={[0.08, 0.08, 0.05, 16]} />
				<meshStandardMaterial color="#222" />
			</mesh>
			<mesh position={[0.15, -CART_HEIGHT / 2, -0.2]} castShadow>
				<cylinderGeometry args={[0.08, 0.08, 0.05, 16]} />
				<meshStandardMaterial color="#222" />
			</mesh>

			{/* Pole pivot point */}
			<mesh position={[0, CART_HEIGHT / 2, 0]} castShadow>
				<sphereGeometry args={[0.08, 16, 16]} />
				<meshStandardMaterial color="#fbbf24" metalness={0.8} roughness={0.2} />
			</mesh>

			{/* Pole */}
			<group rotation={[0, 0, -angle]}>
				<mesh position={[0, POLE_LENGTH / 2, 0]} castShadow receiveShadow>
					<cylinderGeometry
						args={[POLE_RADIUS, POLE_RADIUS, POLE_LENGTH, 16]}
					/>
					<meshStandardMaterial
						color="#10b981"
						metalness={0.4}
						roughness={0.6}
					/>
				</mesh>

				{/* Pole tip */}
				<mesh position={[0, POLE_LENGTH, 0]} castShadow>
					<sphereGeometry args={[POLE_RADIUS * 1.5, 16, 16]} />
					<meshStandardMaterial
						color="#f59e0b"
						metalness={0.6}
						roughness={0.4}
					/>
				</mesh>
			</group>
		</group>
	);
}

export default function CartPole3D({ state }: CartPole3DProps) {
	return (
		<Canvas
			shadows
			camera={{ position: [3, 2, 3], fov: 50 }}
			style={{ background: "#0f172a" }}
		>
			{/* Lighting */}
			<ambientLight intensity={0.4} />
			<directionalLight
				position={[5, 5, 5]}
				intensity={1}
				castShadow
				shadow-mapSize-width={2048}
				shadow-mapSize-height={2048}
			/>
			<pointLight position={[-5, 3, -5]} intensity={0.5} />

			{/* Environment */}
			<Environment preset="city" />

			{/* Grid */}
			<Grid
				args={[10, 10]}
				cellSize={0.5}
				cellThickness={0.5}
				cellColor="#334155"
				sectionSize={2}
				sectionThickness={1}
				sectionColor="#475569"
				fadeDistance={25}
				fadeStrength={1}
				followCamera={false}
				infiniteGrid
			/>

			{/* Scene objects */}
			<Track />
			<Cart position={state.position} angle={state.angle} />

			{/* Camera controls */}
			<OrbitControls
				enablePan
				enableZoom
				enableRotate
				minDistance={2}
				maxDistance={10}
				minPolarAngle={0}
				maxPolarAngle={Math.PI / 2}
			/>
		</Canvas>
	);
}
