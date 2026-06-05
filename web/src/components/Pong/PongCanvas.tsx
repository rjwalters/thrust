import { useEffect, useRef } from "react";
import type { PongState } from "./usePong";

interface PongCanvasProps {
	state: PongState;
}

const PADDLE_W = 12; // pixels
const PADDLE_H_FRAC = 0.2; // fraction of canvas height (matches Rust PADDLE_H*2)
const BALL_R_FRAC = 0.015; // fraction of canvas width

export default function PongCanvas({ state }: PongCanvasProps) {
	const canvasRef = useRef<HTMLCanvasElement>(null);

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;
		const ctx = canvas.getContext("2d");
		if (!ctx) return;

		const W = canvas.width;
		const H = canvas.height;

		// Background
		ctx.fillStyle = "#111827";
		ctx.fillRect(0, 0, W, H);

		// Center dashed line
		ctx.setLineDash([8, 8]);
		ctx.strokeStyle = "rgba(255,255,255,0.15)";
		ctx.lineWidth = 2;
		ctx.beginPath();
		ctx.moveTo(W / 2, 0);
		ctx.lineTo(W / 2, H);
		ctx.stroke();
		ctx.setLineDash([]);

		// Score display
		ctx.fillStyle = "rgba(255,255,255,0.6)";
		ctx.font = `bold ${Math.round(H * 0.12)}px monospace`;
		ctx.textAlign = "center";
		ctx.fillText(String(Math.round(state.leftScore)), W * 0.25, H * 0.15);
		ctx.fillText(String(Math.round(state.rightScore)), W * 0.75, H * 0.15);

		const paddleH = H * PADDLE_H_FRAC;
		const ballR = W * BALL_R_FRAC;

		// Left paddle (agent — blue)
		const leftPX = W * 0.05 - PADDLE_W / 2;
		const leftPY = state.leftY * H - paddleH / 2;
		ctx.fillStyle = "#3b82f6";
		ctx.beginPath();
		ctx.roundRect(leftPX, leftPY, PADDLE_W, paddleH, 4);
		ctx.fill();

		// Right paddle (opponent — red)
		const rightPX = W * 0.95 - PADDLE_W / 2;
		const rightPY = state.rightY * H - paddleH / 2;
		ctx.fillStyle = "#ef4444";
		ctx.beginPath();
		ctx.roundRect(rightPX, rightPY, PADDLE_W, paddleH, 4);
		ctx.fill();

		// Ball (white with glow)
		const ballX = state.ballX * W;
		const ballY = state.ballY * H;

		ctx.shadowColor = "rgba(255,255,255,0.8)";
		ctx.shadowBlur = 12;
		ctx.fillStyle = "#ffffff";
		ctx.beginPath();
		ctx.arc(ballX, ballY, ballR, 0, Math.PI * 2);
		ctx.fill();
		ctx.shadowBlur = 0;

		// Labels
		ctx.font = `${Math.round(H * 0.04)}px sans-serif`;
		ctx.fillStyle = "rgba(59,130,246,0.7)";
		ctx.textAlign = "left";
		ctx.fillText("Agent", W * 0.02, H * 0.97);
		ctx.fillStyle = "rgba(239,68,68,0.7)";
		ctx.textAlign = "right";
		ctx.fillText("Opponent", W * 0.98, H * 0.97);
	}, [state]);

	return (
		<canvas
			ref={canvasRef}
			width={640}
			height={400}
			className="w-full h-full object-contain"
			style={{ background: "#111827" }}
		/>
	);
}
