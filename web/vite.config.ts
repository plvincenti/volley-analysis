import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// En développement, l'API et les vidéos viennent de `volley web` (port 8000).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/media": "http://127.0.0.1:8000",
    },
  },
});
