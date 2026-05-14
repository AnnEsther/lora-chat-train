# ── Stage 1: build static export ─────────────────────────────────────────────
FROM node:20-alpine AS builder
WORKDIR /app

ARG NEXT_PUBLIC_API_URL
ARG NEXT_PUBLIC_API_KEY
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_KEY=$NEXT_PUBLIC_API_KEY

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build
# produces /app/out/

# ── Stage 2: serve with nginx ─────────────────────────────────────────────────
FROM nginx:alpine AS runner
COPY --from=builder /app/out /usr/share/nginx/html/glyph
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
