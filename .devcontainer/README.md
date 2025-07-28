# Development Container Setup

This directory contains the configuration for a VS Code development container that provides a consistent Python development environment using `uv` for fast dependency management.

## Features

- **Python 3.11** environment
- **UV package manager** for fast dependency installation and caching
- **Pre-installed extensions** for Python development, Jupyter notebooks, and more
- **Persistent caching** for UV packages to speed up container rebuilds
- **Virtual environment** automatically activated in all terminals

## What's Included

### Docker Configuration
- `Dockerfile`: Multi-stage build optimized for caching with UV
- `docker-compose.yml`: Development-specific compose configuration
- `.dockerignore`: Optimized for Python projects

### VS Code Configuration
- `devcontainer.json`: Complete devcontainer specification
- Pre-configured Python interpreter path
- Essential Python extensions pre-installed

## Usage

1. **Prerequisites**: 
   - VS Code with the "Dev Containers" extension installed
   - Docker Desktop running

2. **Open in Container**:
   - Open this project in VS Code
   - VS Code should automatically detect the devcontainer configuration
   - Click "Reopen in Container" when prompted
   - Or use Command Palette: `Dev Containers: Reopen in Container`

3. **First Build**:
   - The initial build will take a few minutes as it downloads images and installs dependencies
   - Subsequent builds will be much faster due to caching

## Key Benefits

### UV Caching
- UV cache is persisted in a Docker volume at `/opt/uv-cache`
- Dependencies are cached between container rebuilds
- Much faster than traditional pip installs

### Dependency Management
- Uses `uv.lock` file for reproducible builds
- All dependencies installed with `uv sync --frozen`
- Project installed in development mode with `uv pip install -e .`

### Development Experience
- Virtual environment at `/opt/venv` automatically activated
- Python interpreter pre-configured in VS Code
- All project dependencies available immediately

## Troubleshooting

### Container Won't Start
```bash
# Rebuild the container from scratch
Command Palette > Dev Containers: Rebuild Container
```

### Dependencies Not Installing
```bash
# In the container terminal, manually sync dependencies
uv sync --frozen
```

### Clear All Caches
```bash
# Remove Docker volumes (will require full rebuild)
docker volume rm assignment1-basics_uv-cache assignment1-basics_venv
```

## Customization

### Adding New Dependencies
1. Add dependencies to `pyproject.toml`
2. Run `uv sync` in the container terminal
3. Commit the updated `uv.lock` file

### Installing Additional Extensions
Edit the `extensions` array in `devcontainer.json` and rebuild the container.

### Changing Python Version
Modify the `PYTHON_VERSION` build arg in `docker-compose.yml` and rebuild.
