import { Application } from "@pixi/react";
import type { Application as PixiApplication } from "pixi.js";
import { Container, Graphics } from "pixi.js";
import { useEffect, useRef } from "react";
import type { SnakeState } from "./useSnake";

interface SnakePixiProps {
	state: SnakeState;
}

const CELL_SIZE = 24;
const GRID_PADDING = 20;
const AGENT_COLORS = [
	0x10b981, // emerald
	0x3b82f6, // blue
	0xf59e0b, // amber
	0xec4899, // pink
];

export default function SnakePixi({ state }: SnakePixiProps) {
	const width = state.width * CELL_SIZE + GRID_PADDING * 2;
	const height = state.height * CELL_SIZE + GRID_PADDING * 2;
	const containerRef = useRef<Container | null>(null);
	const gridGraphicsRef = useRef<Graphics | null>(null);
	const foodGraphicsRef = useRef<Graphics | null>(null);
	const snakesGraphicsRef = useRef<Graphics | null>(null);

	const onInit = (app: PixiApplication) => {
		// Create container and graphics objects
		const container = new Container();
		const gridGraphics = new Graphics();
		const foodGraphics = new Graphics();
		const snakesGraphics = new Graphics();

		container.addChild(gridGraphics);
		container.addChild(foodGraphics);
		container.addChild(snakesGraphics);
		app.stage.addChild(container);

		containerRef.current = container;
		gridGraphicsRef.current = gridGraphics;
		foodGraphicsRef.current = foodGraphics;
		snakesGraphicsRef.current = snakesGraphics;

		// Draw initial grid
		drawGrid(gridGraphics, state);
	};

	useEffect(() => {
		if (!foodGraphicsRef.current || !snakesGraphicsRef.current) return;

		drawFood(foodGraphicsRef.current, state);
		drawSnakes(snakesGraphicsRef.current, state);
	}, [state]);

	const drawGrid = (g: Graphics, st: SnakeState) => {
		g.clear();

		// Background
		g.rect(0, 0, width, height);
		g.fill(0x1a1a2e);

		// Grid lines
		for (let x = 0; x <= st.width; x++) {
			const xPos = GRID_PADDING + x * CELL_SIZE;
			g.moveTo(xPos, GRID_PADDING);
			g.lineTo(xPos, height - GRID_PADDING);
			g.stroke({ width: 1, color: 0x2a2a3e, alpha: 0.3 });
		}

		for (let y = 0; y <= st.height; y++) {
			const yPos = GRID_PADDING + y * CELL_SIZE;
			g.moveTo(GRID_PADDING, yPos);
			g.lineTo(width - GRID_PADDING, yPos);
			g.stroke({ width: 1, color: 0x2a2a3e, alpha: 0.3 });
		}
	};

	const drawFood = (g: Graphics, st: SnakeState) => {
		g.clear();

		for (let i = 0; i < st.foodPositions.length; i += 2) {
			const x = st.foodPositions[i];
			const y = st.foodPositions[i + 1];

			const centerX = GRID_PADDING + x * CELL_SIZE + CELL_SIZE / 2;
			const centerY = GRID_PADDING + y * CELL_SIZE + CELL_SIZE / 2;

			// Glow effect
			g.circle(centerX, centerY, 8);
			g.fill({ color: 0xff6b6b, alpha: 0.3 });

			// Core
			g.circle(centerX, centerY, 5);
			g.fill(0xff6b6b);
		}
	};

	const drawSnakes = (g: Graphics, st: SnakeState) => {
		g.clear();

		// Parse snake positions
		// Format: [num_snakes, len0, x0, y0, x1, y1, ..., len1, x0, y0, ...]
		if (st.snakePositions.length === 0) return;

		const numSnakes = st.snakePositions[0];
		let idx = 1;

		for (let agentId = 0; agentId < numSnakes; agentId++) {
			if (idx >= st.snakePositions.length) break;

			const length = st.snakePositions[idx];
			idx++;

			if (length <= 0) continue;

			const isActive = st.activeAgents[agentId] === 1;
			const color = AGENT_COLORS[agentId % AGENT_COLORS.length];
			const alpha = isActive ? 1.0 : 0.3;

			// Draw segments
			for (let i = 0; i < length; i++) {
				if (idx + i * 2 + 1 >= st.snakePositions.length) break;

				const x = st.snakePositions[idx + i * 2];
				const y = st.snakePositions[idx + i * 2 + 1];

				const centerX = GRID_PADDING + x * CELL_SIZE + CELL_SIZE / 2;
				const centerY = GRID_PADDING + y * CELL_SIZE + CELL_SIZE / 2;

				const isHead = i === 0; // First segment is the head
				const segmentSize = isHead ? 10 : 9;

				// Segment glow
				if (isActive) {
					g.roundRect(
						centerX - segmentSize - 2,
						centerY - segmentSize - 2,
						(segmentSize + 2) * 2,
						(segmentSize + 2) * 2,
						4,
					);
					g.fill({ color, alpha: 0.3 });
				}

				// Segment body
				g.roundRect(
					centerX - segmentSize,
					centerY - segmentSize,
					segmentSize * 2,
					segmentSize * 2,
					3,
				);
				g.fill({ color, alpha });

				// Head eyes
				if (isHead && isActive) {
					const eyeSize = 2;
					const eyeOffset = 3;

					g.circle(centerX - eyeOffset, centerY - eyeOffset, eyeSize);
					g.fill(0xffffff);

					g.circle(centerX + eyeOffset, centerY - eyeOffset, eyeSize);
					g.fill(0xffffff);
				}
			}

			idx += length * 2;
		}
	};

	return (
		<Application
			width={width}
			height={height}
			backgroundAlpha={1}
			backgroundColor={0x1a1a2e}
			antialias
			onInit={onInit}
		/>
	);
}
