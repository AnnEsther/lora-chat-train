import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  output: 'export',
  basePath: '/glyph',      // served at train.anratelier.com/glyph
  assetPrefix: '/glyph',   // all /_next/static/... urls prefixed
  images: {
    unoptimized: true, // required for static export
  },
};

export default nextConfig;
