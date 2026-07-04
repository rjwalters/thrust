import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
	CartesianGrid,
	Legend,
	Line,
	LineChart,
	ResponsiveContainer,
	Tooltip,
	XAxis,
	YAxis,
} from "recharts";
import Footer from "../components/Footer";

// One row of the merged learning-curve dataset. `step` is the shared x-axis
// (env steps); each algorithm column holds the mean episode reward recorded at
// that step, or `undefined` when that algorithm did not log at that exact step.
interface CurvePoint {
	step: number;
	a2c?: number;
	dqn?: number;
	ppo?: number;
}

type Algorithm = "a2c" | "dqn" | "ppo";

interface SeriesConfig {
	key: Algorithm;
	label: string;
	color: string;
	file: string;
}

// The three CartPole examples share the same environment, seed (0) and step
// budget (60k env steps), so their curves overlay directly on one chart.
const SERIES: SeriesConfig[] = [
	{ key: "a2c", label: "A2C", color: "#6366f1", file: "cartpole_curves_a2c.csv" },
	{ key: "dqn", label: "DQN", color: "#10b981", file: "cartpole_curves_dqn.csv" },
	{ key: "ppo", label: "PPO", color: "#f59e0b", file: "cartpole_curves_ppo.csv" },
];

// Parse a `env_steps,mean_episode_reward` CSV (header row first) into
// [step, reward] tuples. Blank lines are skipped and malformed rows ignored.
function parseCurveCsv(text: string): Array<[number, number]> {
	return text
		.trim()
		.split("\n")
		.slice(1) // drop the `env_steps,mean_episode_reward` header
		.map((line) => line.split(","))
		.filter((cols) => cols.length >= 2)
		.map(([steps, reward]) => [Number(steps), Number(reward)] as [number, number])
		.filter(([steps, reward]) => Number.isFinite(steps) && Number.isFinite(reward));
}

export default function TrainingDashboardPage() {
	const [data, setData] = useState<CurvePoint[] | null>(null);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		let cancelled = false;

		async function loadCurves() {
			try {
				const responses = await Promise.all(
					SERIES.map(async (series) => {
						const res = await fetch(`${import.meta.env.BASE_URL}${series.file}`);
						if (!res.ok) {
							throw new Error(`Failed to load ${series.file}: ${res.status}`);
						}
						return { key: series.key, rows: parseCurveCsv(await res.text()) };
					}),
				);

				// Merge the three curves into one array keyed by env step, so a
				// single recharts <LineChart> can render all three <Line> series.
				const byStep = new Map<number, CurvePoint>();
				for (const { key, rows } of responses) {
					for (const [step, reward] of rows) {
						const point = byStep.get(step) ?? { step };
						point[key] = reward;
						byStep.set(step, point);
					}
				}
				const merged = [...byStep.values()].sort((a, b) => a.step - b.step);

				if (!cancelled) {
					setData(merged);
				}
			} catch (err) {
				console.error("Failed to load training curves:", err);
				if (!cancelled) {
					setError(err instanceof Error ? err.message : String(err));
				}
			}
		}

		loadCurves();
		return () => {
			cancelled = true;
		};
	}, []);

	return (
		<div className="min-h-screen bg-gray-50">
			<div className="container mx-auto px-4 py-8">
				<Link
					to="/"
					className="text-indigo-600 hover:text-indigo-800 mb-4 inline-block"
				>
					← Back to Home
				</Link>
				<h1 className="text-4xl font-bold mb-2">Training Dashboard</h1>
				<p className="text-gray-600 mb-8 max-w-3xl">
					Watch three reinforcement-learning algorithms learn to balance the
					CartPole. Each curve is the mean episode reward (episode length)
					plotted against environment steps as training progresses.
				</p>

				{/* Chart */}
				<div className="bg-white rounded-lg shadow-lg p-6">
					<h2 className="text-2xl font-bold mb-4">
						CartPole-v1 — Learning Curves
					</h2>

					{error ? (
						<div className="flex items-center justify-center h-[480px] text-red-600">
							Failed to load training curves: {error}
						</div>
					) : !data ? (
						<div className="flex items-center justify-center h-[480px] text-gray-500">
							Loading training curves…
						</div>
					) : (
						<div className="w-full h-[480px]">
							<ResponsiveContainer width="100%" height="100%">
								<LineChart
									data={data}
									margin={{ top: 8, right: 24, bottom: 24, left: 8 }}
								>
									<CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
									<XAxis
										dataKey="step"
										type="number"
										domain={["dataMin", "dataMax"]}
										tickFormatter={(v) => `${Math.round(Number(v) / 1000)}k`}
										label={{
											value: "environment steps",
											position: "insideBottom",
											offset: -12,
										}}
									/>
									<YAxis
										label={{
											value: "mean episode reward",
											angle: -90,
											position: "insideLeft",
											style: { textAnchor: "middle" },
										}}
									/>
									<Tooltip
										formatter={(value, name) => [
											Number(value).toFixed(1),
											String(name),
										]}
										labelFormatter={(label) =>
											`${Number(label).toLocaleString()} steps`
										}
									/>
									<Legend />
									{SERIES.map((series) => (
										<Line
											key={series.key}
											type="monotone"
											dataKey={series.key}
											name={series.label}
											stroke={series.color}
											dot={false}
											strokeWidth={2}
											connectNulls
											isAnimationActive={false}
										/>
									))}
								</LineChart>
							</ResponsiveContainer>
						</div>
					)}

					<p className="text-sm text-gray-500 mt-4">
						<strong>Data source:</strong> replay of recorded training runs —
						CartPole-v1, seed 0, 60k environment steps per algorithm. Curves were
						generated locally by the repository's training examples
						(<code>train_cartpole_a2c</code>, <code>train_cartpole_dqn</code>,{" "}
						<code>train_cartpole_modern</code>) via the <code>CURVE_CSV</code>{" "}
						environment variable and committed as static CSV files under{" "}
						<code>web/public/</code>. No training runs in your browser on this
						page.
					</p>
				</div>

				{/* Explanation */}
				<div className="mt-8 bg-white rounded-lg shadow-lg p-6">
					<h2 className="text-2xl font-bold mb-4">About These Algorithms</h2>
					<p className="text-gray-600 mb-6">
						All three agents optimize the same objective — keep the pole
						balanced for as long as possible, earning +1 reward per timestep —
						but they get there very differently. Because they share the same
						environment, random seed and step budget, their curves are directly
						comparable.
					</p>

					<div className="grid md:grid-cols-3 gap-6">
						<div>
							<h3 className="font-semibold mb-2 flex items-center gap-2">
								<span
									className="inline-block w-3 h-3 rounded-full"
									style={{ backgroundColor: "#6366f1" }}
								/>
								A2C
							</h3>
							<p className="text-sm text-gray-600">
								Advantage Actor-Critic — a synchronous on-policy method that
								updates a shared policy and value network from short rollouts.
								Simple and fast, but higher variance than PPO.
							</p>
						</div>

						<div>
							<h3 className="font-semibold mb-2 flex items-center gap-2">
								<span
									className="inline-block w-3 h-3 rounded-full"
									style={{ backgroundColor: "#10b981" }}
								/>
								DQN
							</h3>
							<p className="text-sm text-gray-600">
								Deep Q-Network — an off-policy value-based method that learns an
								action-value function from a replay buffer, acting
								ε-greedily. Sample-efficient on discrete-action tasks like
								CartPole.
							</p>
						</div>

						<div>
							<h3 className="font-semibold mb-2 flex items-center gap-2">
								<span
									className="inline-block w-3 h-3 rounded-full"
									style={{ backgroundColor: "#f59e0b" }}
								/>
								PPO
							</h3>
							<p className="text-sm text-gray-600">
								Proximal Policy Optimization — an on-policy method that clips
								policy updates to stay close to the previous policy. Stable and
								widely used; this is the algorithm behind the CartPole inference
								demo.
							</p>
						</div>
					</div>

					<div className="mt-6 p-4 bg-blue-50 rounded-lg">
						<p className="text-sm text-blue-900">
							<strong>What you're seeing:</strong> the y-axis is the mean episode
							length (each surviving timestep is +1 reward), capped at 500 for a
							solved CartPole. A rising curve means the agent is learning to keep
							the pole up for longer. Convergence speed and stability differ by
							algorithm.
						</p>
					</div>
				</div>

				<Footer />
			</div>
		</div>
	);
}
