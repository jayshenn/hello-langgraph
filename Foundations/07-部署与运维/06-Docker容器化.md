# 06-Docker容器化

## 概述

Docker 容器化为 LangGraph 应用提供一致的运行环境，支持快速部署、版本管理和横向扩展。本文档涵盖从基础容器构建到生产级 Kubernetes 部署的完整方案。

## 基础容器化

### Dockerfile 最佳实践
```dockerfile
# Dockerfile
FROM python:3.11-slim as base

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 创建非root用户
RUN groupadd -r langgraph && useradd -r -g langgraph langgraph

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt pyproject.toml ./

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY src/ ./src/
COPY langgraph.json ./

# 切换到非root用户
USER langgraph

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "langgraph", "run", "--host", "0.0.0.0", "--port", "8000"]
```

### 多阶段构建
```dockerfile
# Dockerfile.multistage
# 构建阶段
FROM python:3.11-slim as builder

ENV PIP_NO_CACHE_DIR=1
WORKDIR /build

# 安装构建依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖到临时目录
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# 运行阶段
FROM python:3.11-slim as runtime

# 创建用户
RUN groupadd -r langgraph && useradd -r -g langgraph langgraph

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制Python包
COPY --from=builder /root/.local /home/langgraph/.local

# 设置PATH
ENV PATH=/home/langgraph/.local/bin:$PATH

WORKDIR /app

# 复制应用代码
COPY --chown=langgraph:langgraph src/ ./src/
COPY --chown=langgraph:langgraph langgraph.json ./

USER langgraph

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "langgraph", "run", "--host", "0.0.0.0"]
```

## Docker Compose 部署

### 完整服务栈
```yaml
# docker-compose.yml
version: '3.8'

services:
  # PostgreSQL 数据库
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-langgraph}
      POSTGRES_USER: ${POSTGRES_USER:-langgraph}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-langgraph}"]
      interval: 30s
      timeout: 10s
      retries: 5
    restart: unless-stopped

  # Redis 缓存
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  # LangGraph 应用
  langgraph-app:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - POSTGRES_URL=postgresql://${POSTGRES_USER:-langgraph}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB:-langgraph}
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LANGCHAIN_TRACING_V2=${LANGCHAIN_TRACING_V2:-false}
      - LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}
      - LANGCHAIN_PROJECT=${LANGCHAIN_PROJECT:-langgraph-local}
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1'
        reservations:
          memory: 1G
          cpus: '0.5'

  # Nginx 负载均衡器
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - langgraph-app
    restart: unless-stopped

  # Prometheus 监控
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  # Grafana 仪表板
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    driver: bridge
```

### 环境配置
```bash
# .env
# 数据库配置
POSTGRES_DB=langgraph
POSTGRES_USER=langgraph
POSTGRES_PASSWORD=your_secure_password_here

# API密钥
OPENAI_API_KEY=sk-your_openai_api_key_here

# LangSmith配置
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=langgraph-production

# 监控配置
GRAFANA_PASSWORD=admin123
```

## Kubernetes 部署

### 命名空间和配置
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: langgraph
  labels:
    name: langgraph

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: langgraph-config
  namespace: langgraph
data:
  langgraph.json: |
    {
      "dependencies": ["."],
      "graphs": {
        "main_agent": "./src/agent/graph.py:graph"
      },
      "env": ".env",
      "python_version": "3.11"
    }

---
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: langgraph-secrets
  namespace: langgraph
type: Opaque
stringData:
  postgres-password: "your_secure_password"
  openai-api-key: "sk-your_openai_api_key"
  langsmith-api-key: "your_langsmith_api_key"
```

### 数据库部署
```yaml
# postgres.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: langgraph
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: "langgraph"
        - name: POSTGRES_USER
          value: "langgraph"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: langgraph-secrets
              key: postgres-password
        - name: PGDATA
          value: "/var/lib/postgresql/data/pgdata"
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        livenessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - langgraph
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - langgraph
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: "gp2"
      resources:
        requests:
          storage: 20Gi

---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: langgraph
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
```

### 应用部署
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langgraph-app
  namespace: langgraph
  labels:
    app: langgraph-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langgraph-app
  template:
    metadata:
      labels:
        app: langgraph-app
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: langgraph-app
        image: langgraph/my-agent:latest
        imagePullPolicy: Always
        env:
        - name: POSTGRES_URL
          value: "postgresql://langgraph:$(POSTGRES_PASSWORD)@postgres:5432/langgraph"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: langgraph-secrets
              key: postgres-password
        - name: REDIS_URL
          value: "redis://redis:6379"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: langgraph-secrets
              key: openai-api-key
        - name: LANGCHAIN_TRACING_V2
          value: "true"
        - name: LANGCHAIN_API_KEY
          valueFrom:
            secretKeyRef:
              name: langgraph-secrets
              key: langsmith-api-key
        - name: LANGCHAIN_PROJECT
          value: "langgraph-k8s"
        ports:
        - containerPort: 8000
          name: http
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: config-volume
        configMap:
          name: langgraph-config
      - name: logs-volume
        emptyDir: {}
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - langgraph-app
              topologyKey: kubernetes.io/hostname

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: langgraph-service
  namespace: langgraph
  labels:
    app: langgraph-app
spec:
  selector:
    app: langgraph-app
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  type: ClusterIP

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: langgraph-ingress
  namespace: langgraph
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.langgraph.example.com
    secretName: langgraph-tls
  rules:
  - host: api.langgraph.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: langgraph-service
            port:
              number: 80
```

## 容器优化

### 镜像优化
```dockerfile
# Dockerfile.optimized
# 使用Alpine Linux减小镜像大小
FROM python:3.11-alpine as builder

# 安装构建依赖
RUN apk add --no-cache \
    gcc \
    musl-dev \
    libffi-dev \
    openssl-dev \
    cargo \
    rust

WORKDIR /build

# 安装Python依赖
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# 运行时镜像
FROM python:3.11-alpine

# 安装运行时依赖
RUN apk add --no-cache \
    curl \
    ca-certificates \
    && addgroup -g 1000 langgraph \
    && adduser -D -s /bin/sh -u 1000 -G langgraph langgraph

# 复制wheels并安装
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache /wheels/*

WORKDIR /app

# 复制应用代码
COPY --chown=langgraph:langgraph src/ ./src/
COPY --chown=langgraph:langgraph langgraph.json ./

USER langgraph

# 精简的健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "langgraph", "run", "--host", "0.0.0.0"]
```

### 安全加固
```dockerfile
# Dockerfile.secure
FROM python:3.11-slim

# 创建非特权用户
RUN groupadd -r -g 999 langgraph && \
    useradd -r -g langgraph -u 999 langgraph

# 安装必要的系统包
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get purge -y --auto-remove

# 设置安全的工作目录
RUN mkdir -p /app && \
    chown -R langgraph:langgraph /app

WORKDIR /app

# 复制和安装依赖
COPY --chown=langgraph:langgraph requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 复制应用代码
COPY --chown=langgraph:langgraph src/ ./src/
COPY --chown=langgraph:langgraph langgraph.json ./

# 切换到非特权用户
USER langgraph

# 移除潜在的危险功能
RUN rm -rf /tmp/* /var/tmp/*

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 只暴露必要的端口
EXPOSE 8000

# 使用exec格式避免shell注入
CMD ["python", "-m", "langgraph", "run", "--host", "0.0.0.0", "--port", "8000"]
```

## 容器编排

### Docker Swarm 部署
```yaml
# docker-stack.yml
version: '3.8'

services:
  langgraph-app:
    image: langgraph/my-agent:latest
    ports:
      - "8000:8000"
    environment:
      - POSTGRES_URL=postgresql://langgraph:${POSTGRES_PASSWORD}@postgres:5432/langgraph
      - REDIS_URL=redis://redis:6379
    networks:
      - langgraph-network
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      update_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
      rollback_config:
        parallelism: 1
        delay: 10s
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'
      placement:
        constraints:
          - node.role == worker

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: langgraph
      POSTGRES_USER: langgraph
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - langgraph-network
    secrets:
      - postgres_password
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.role == manager

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - langgraph-network
    deploy:
      replicas: 1

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    configs:
      - source: nginx_config
        target: /etc/nginx/nginx.conf
    networks:
      - langgraph-network
    deploy:
      replicas: 2
      placement:
        constraints:
          - node.role == manager

volumes:
  postgres_data:
  redis_data:

networks:
  langgraph-network:
    driver: overlay
    attachable: true

secrets:
  postgres_password:
    external: true

configs:
  nginx_config:
    external: true
```

### Helm Chart 部署
```yaml
# charts/langgraph/Chart.yaml
apiVersion: v2
name: langgraph
description: A Helm chart for LangGraph applications
type: application
version: 0.1.0
appVersion: "1.0.0"

---
# charts/langgraph/values.yaml
replicaCount: 3

image:
  repository: langgraph/my-agent
  pullPolicy: IfNotPresent
  tag: "latest"

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: api.langgraph.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: langgraph-tls
      hosts:
        - api.langgraph.example.com

resources:
  limits:
    cpu: 500m
    memory: 1Gi
  requests:
    cpu: 250m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

postgresql:
  enabled: true
  auth:
    postgresPassword: "changeme"
    username: "langgraph"
    password: "changeme"
    database: "langgraph"

redis:
  enabled: true
  auth:
    enabled: false

---
# charts/langgraph/templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "langgraph.fullname" . }}
  labels:
    {{- include "langgraph.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "langgraph.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "langgraph.selectorLabels" . | nindent 8 }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: 8000
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          env:
            - name: POSTGRES_URL
              value: "postgresql://{{ .Values.postgresql.auth.username }}:{{ .Values.postgresql.auth.password }}@{{ include "langgraph.fullname" . }}-postgresql:5432/{{ .Values.postgresql.auth.database }}"
            - name: REDIS_URL
              value: "redis://{{ include "langgraph.fullname" . }}-redis-master:6379"
```

## 部署脚本

### 自动化部署脚本
```bash
#!/bin/bash
# deploy.sh

set -euo pipefail

# 配置变量
DOCKER_REGISTRY="your-registry.com"
IMAGE_NAME="langgraph/my-agent"
VERSION="${1:-latest}"
ENVIRONMENT="${2:-production}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# 检查依赖
check_dependencies() {
    log "检查部署依赖..."

    command -v docker >/dev/null 2>&1 || error "Docker 未安装"
    command -v kubectl >/dev/null 2>&1 || error "kubectl 未安装"
    command -v helm >/dev/null 2>&1 || warn "Helm 未安装，将跳过 Helm 部署"

    log "依赖检查完成"
}

# 构建镜像
build_image() {
    log "构建 Docker 镜像..."

    docker build \
        -t "${IMAGE_NAME}:${VERSION}" \
        -t "${IMAGE_NAME}:latest" \
        --build-arg VERSION="${VERSION}" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        .

    log "镜像构建完成: ${IMAGE_NAME}:${VERSION}"
}

# 推送镜像
push_image() {
    log "推送镜像到仓库..."

    docker tag "${IMAGE_NAME}:${VERSION}" "${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}"
    docker tag "${IMAGE_NAME}:${VERSION}" "${DOCKER_REGISTRY}/${IMAGE_NAME}:latest"

    docker push "${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}"
    docker push "${DOCKER_REGISTRY}/${IMAGE_NAME}:latest"

    log "镜像推送完成"
}

# 部署到Kubernetes
deploy_kubernetes() {
    log "部署到 Kubernetes..."

    # 检查集群连接
    kubectl cluster-info || error "无法连接到 Kubernetes 集群"

    # 创建命名空间
    kubectl create namespace langgraph --dry-run=client -o yaml | kubectl apply -f -

    # 部署应用
    kubectl apply -f k8s/ -n langgraph

    # 等待部署完成
    kubectl rollout status deployment/langgraph-app -n langgraph --timeout=300s

    log "Kubernetes 部署完成"
}

# 部署到Docker Swarm
deploy_swarm() {
    log "部署到 Docker Swarm..."

    # 检查Swarm状态
    docker node ls >/dev/null 2>&1 || error "Docker Swarm 未初始化"

    # 部署服务栈
    docker stack deploy -c docker-stack.yml langgraph

    log "Docker Swarm 部署完成"
}

# Helm部署
deploy_helm() {
    if ! command -v helm >/dev/null 2>&1; then
        warn "Helm 未安装，跳过 Helm 部署"
        return
    fi

    log "使用 Helm 部署..."

    # 更新依赖
    helm dependency update charts/langgraph/

    # 部署或升级
    helm upgrade --install langgraph charts/langgraph/ \
        --namespace langgraph \
        --create-namespace \
        --set image.tag="${VERSION}" \
        --set environment="${ENVIRONMENT}" \
        --wait \
        --timeout=300s

    log "Helm 部署完成"
}

# 健康检查
health_check() {
    log "执行健康检查..."

    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if kubectl get pods -n langgraph | grep -q "Running"; then
            log "应用启动成功"
            return 0
        fi

        warn "等待应用启动... (尝试 $attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done

    error "健康检查失败，应用未能在预期时间内启动"
}

# 主函数
main() {
    log "开始部署 LangGraph 应用..."
    log "版本: ${VERSION}"
    log "环境: ${ENVIRONMENT}"

    check_dependencies
    build_image

    if [ "${ENVIRONMENT}" = "production" ]; then
        push_image
    fi

    case "${DEPLOY_TYPE:-kubernetes}" in
        kubernetes)
            deploy_kubernetes
            ;;
        swarm)
            deploy_swarm
            ;;
        helm)
            deploy_helm
            ;;
        *)
            error "不支持的部署类型: ${DEPLOY_TYPE}"
            ;;
    esac

    health_check

    log "部署完成！"
    log "访问地址: https://api.langgraph.example.com"
}

# 清理函数
cleanup() {
    log "清理部署环境..."

    case "${DEPLOY_TYPE:-kubernetes}" in
        kubernetes)
            kubectl delete namespace langgraph --ignore-not-found=true
            ;;
        swarm)
            docker stack rm langgraph
            ;;
        helm)
            helm uninstall langgraph -n langgraph || true
            kubectl delete namespace langgraph --ignore-not-found=true
            ;;
    esac

    log "清理完成"
}

# 脚本参数处理
case "${3:-deploy}" in
    deploy)
        main
        ;;
    cleanup)
        cleanup
        ;;
    *)
        error "用法: $0 <version> <environment> [deploy|cleanup]"
        ;;
esac
```

## 故障排除

### 常见容器问题
```bash
# 容器启动失败
docker logs <container_id>

# 进入容器调试
docker exec -it <container_id> /bin/sh

# 检查资源使用
docker stats <container_id>

# 检查网络连接
docker network ls
docker network inspect <network_name>
```

### Kubernetes 调试
```bash
# 查看Pod状态
kubectl get pods -n langgraph -o wide

# 查看Pod日志
kubectl logs -f deployment/langgraph-app -n langgraph

# 进入Pod调试
kubectl exec -it <pod_name> -n langgraph -- /bin/sh

# 查看事件
kubectl get events -n langgraph --sort-by='.lastTimestamp'

# 描述资源
kubectl describe pod <pod_name> -n langgraph
```

## 下一步

- 🔐 学习 [07-认证与授权](./07-认证与授权.md) - 安全访问控制
- 🔗 了解 [08-Webhooks集成](./08-Webhooks集成.md) - 事件驱动架构

## 相关链接

- [Docker 最佳实践](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes 部署指南](https://kubernetes.io/docs/concepts/workloads/deployment/)
- [Helm Chart 开发](https://helm.sh/docs/chart_template_guide/)
- [容器安全指南](https://kubernetes.io/docs/concepts/security/)