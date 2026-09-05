# Build the bundle, then serve it from nginx alongside a proxy to the API.
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci || npm install
COPY tsconfig.json vite.config.ts index.html ./
COPY src ./src
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY proxy_common.conf /etc/nginx/proxy_common.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
