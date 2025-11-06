import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";
import { execSync } from "child_process";

// Get git commit hash and build timestamp
const commitHash = execSync("git rev-parse --short HEAD").toString().trim();
const buildTime = new Date().toISOString();

// https://vite.dev/config/
export default defineConfig({
	plugins: [react(), tailwindcss()],
	base: "/thrust/",
	define: {
		__COMMIT_HASH__: JSON.stringify(commitHash),
		__BUILD_TIME__: JSON.stringify(buildTime),
	},
	build: {
		outDir: "dist",
		assetsDir: "assets",
		sourcemap: false,
		rollupOptions: {
			output: {
				manualChunks: {
					three: ["three", "@react-three/fiber", "@react-three/drei"],
					pixi: ["pixi.js", "@pixi/react"],
				},
			},
		},
	},
});
