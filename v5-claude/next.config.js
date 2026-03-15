/** @type {import('next').NextConfig} */
const nextConfig = {
  // Required for react-pdf worker
  webpack: (config) => {
    config.resolve.alias.canvas = false;
    config.resolve.alias.encoding = false;
    return config;
  },
};

module.exports = nextConfig;
