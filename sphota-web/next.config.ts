import type { NextConfig } from "next";

// The Python service (python -m truesight.app.api). The browser only ever talks to this origin, so images, the .sphota download
// and the event stream need no CORS and work from any host that serves the site.
const API = process.env.SPHOTA_API ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  // The rewrite proxy gives up after 30 s by default. The model API serves one photo at a time and a CPU decode takes minutes, so a
  // second upload (or Compare while a decode runs) waits for its turn: without this it failed with a 500 and the page showed an error.
  experimental: { proxyTimeout: 900_000 },
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
