# Multi-stage build for Rust inference engine
FROM rust:1.75-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY inference-engine/Cargo.toml ./inference-engine/
COPY feature-pipeline/Cargo.toml ./feature-pipeline/
COPY ldc-engine/Cargo.toml ./ldc-engine/
COPY signal-fusion/Cargo.toml ./signal-fusion/

# Create dummy source files for dependency resolution
RUN mkdir -p inference-engine/src feature-pipeline/src ldc-engine/src signal-fusion/src
RUN echo "fn main() {}" > inference-engine/src/main.rs
RUN echo "" > feature-pipeline/src/lib.rs
RUN echo "" > ldc-engine/src/lib.rs
RUN echo "" > signal-fusion/src/lib.rs

# Build dependencies
RUN cargo build --release

# Copy actual source code
COPY . .

# Build the application
RUN cargo build --release --bin inference-engine

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -s /bin/false imp

# Copy binary from builder stage
COPY --from=builder /app/target/release/inference-engine /usr/local/bin/

# Set ownership and permissions
RUN chown imp:imp /usr/local/bin/inference-engine
RUN chmod +x /usr/local/bin/inference-engine

# Switch to non-root user
USER imp

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["inference-engine"]
