# ===========================================================================
# MedVision AI — production image (multi-stage)
#   docker build -t medvision-ai .
#   docker run -p 3000:3000 -e JWT_SECRET=... -e GEMINI_API_KEY=... medvision-ai
# ===========================================================================

# --- build stage ------------------------------------------------------------
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build   # vite build (frontend) + esbuild bundle (dist/server.cjs)

# --- runtime stage ----------------------------------------------------------
FROM node:20-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app/package.json ./package.json
COPY --from=build /app/package-lock.json ./package-lock.json
COPY --from=build /app/dist ./dist
# Production dependencies only. `vite` is declared in `dependencies` (not just
# devDependencies) because dist/server.cjs requires it as an external package.
RUN npm ci --omit=dev && npm cache clean --force

EXPOSE 3000
CMD ["node", "dist/server.cjs"]
