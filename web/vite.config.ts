import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Compilé dans le paquet Python (servi par FastAPI) ; en développement, l'API et les
// images sont relayées vers « champ serveur » (port 8000 par défaut, CHAMP_API sinon).
const api = process.env.CHAMP_API ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  build: { outDir: "../champ_dhonneur/server/web", emptyOutDir: true, chunkSizeWarningLimit: 800 },
  server: { proxy: { "/api": api, "/img": api } },
});
