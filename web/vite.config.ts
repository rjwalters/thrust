import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// https://vite.dev/config/
export default defineConfig({
	plugins: [react()],
	base: "/thrust/",
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
