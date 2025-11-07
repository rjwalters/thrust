import React from "react";

type HouseState = 0 | 1 | 2; // SAFE = 0, BURNING = 1, RUINED = 2

interface TownProps {
	houses: HouseState[];
	numAgents?: number;
	archetypes?: string[];
	className?: string;
}

const Town: React.FC<TownProps> = ({
	houses,
	numAgents = 4,
	archetypes,
	className = "",
}) => {
	// Create positions for 10 houses in a circle
	const housePositions = Array.from({ length: 10 }, (_, i) => {
		const angle = (i / 10) * 2 * Math.PI - Math.PI / 2; // Start from top
		const radius = 231; // Distance from center
		const centerX = 320; // Center of the circle
		const centerY = 320; // Center of the circle

		return {
			x: centerX + radius * Math.cos(angle),
			y: centerY + radius * Math.sin(angle),
			index: i,
			angle: angle,
		};
	});

	const getHouseClass = (state: HouseState) => {
		switch (state) {
			case 0:
				return "house-safe"; // SAFE
			case 1:
				return "house-burning"; // BURNING
			case 2:
				return "house-ruined"; // RUINED
			default:
				return "house-ruined";
		}
	};

	const getHouseSymbol = (state: HouseState) => {
		switch (state) {
			case 0:
				return "🏠"; // SAFE
			case 1:
				return "🔥"; // BURNING
			case 2:
				return "💀"; // RUINED
			default:
				return "❓";
		}
	};

	return (
		<div className={`town-visualization ${className}`}>
			<svg
				width="640"
				height="640"
				viewBox="0 0 640 640"
				className="town-svg absolute inset-0"
			>
				{/* Connection lines between houses */}
				{housePositions.map((pos, i) => {
					const nextPos = housePositions[(i + 1) % 10];
					return (
						<line
							key={`connection-${i}`}
							x1={pos.x}
							y1={pos.y}
							x2={nextPos.x}
							y2={nextPos.y}
							stroke="#e5e7eb"
							strokeWidth="2"
							className="town-connection"
						/>
					);
				})}

				{/* Houses */}
				{housePositions.map((pos, i) => {
					const houseState = houses[i];
					const symbol = getHouseSymbol(houseState);
					const owner = i % numAgents;
					const archetype = archetypes?.[owner];
					const ownerLabel = archetype ? `${archetype} ${owner}` : `Agent ${owner}`;

					return (
						<g key={`house-${i}`} className="town-house-group">
							{/* House circle */}
							<circle
								cx={pos.x}
								cy={pos.y}
								r="50"
								className={`town-house ${getHouseClass(houseState)}`}
								data-house-index={i}
								data-house-state={houseState}
								fill={
									houseState === 0
										? "#86efac"
										: houseState === 1
											? "#fbbf24"
											: "#ef4444"
								}
								stroke={
									houseState === 0
										? "#22c55e"
										: houseState === 1
											? "#f59e0b"
											: "#dc2626"
								}
								strokeWidth="2"
							/>

							{/* House symbol */}
							<text
								x={pos.x}
								y={pos.y}
								textAnchor="middle"
								dominantBaseline="middle"
								className="town-house-symbol select-none"
								style={{ fontSize: "2rem" }}
							>
								{symbol}
							</text>

							{/* Owner index */}
							<text
								x={pos.x}
								y={pos.y + 65}
								textAnchor="middle"
								dominantBaseline="middle"
								className="town-owner-index select-none text-gray-600 dark:text-gray-400"
								style={{ fontSize: "0.75rem", fontWeight: "bold" }}
							>
								{ownerLabel}
							</text>
						</g>
					);
				})}
			</svg>

			{/* Legend */}
			<div className="town-legend mt-4 flex flex-wrap gap-3 justify-center text-sm">
				<div className="flex items-center space-x-2">
					<div className="w-4 h-4 rounded-full bg-green-300 border-2 border-green-500" />
					<span>Safe (🏠)</span>
				</div>
				<div className="flex items-center space-x-2">
					<span className="text-2xl">🔥</span>
					<span>Burning</span>
				</div>
				<div className="flex items-center space-x-2">
					<span className="text-2xl">💀</span>
					<span>Ruined</span>
				</div>
			</div>
		</div>
	);
};

export default Town;
