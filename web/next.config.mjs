/** @type {import('next').NextConfig} */
const nextConfig = {
  // The pipeline server is a separate process (pipeline/server/app.py) holding
  // ~7 GB of models, so it is never bundled here. The browser talks to it
  // directly over WebSocket; NEXT_PUBLIC_PIPELINE_URL points at it.
  reactStrictMode: true,
};

export default nextConfig;
