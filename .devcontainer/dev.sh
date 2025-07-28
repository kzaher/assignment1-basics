#!/bin/bash

# CS336 Assignment 1 - Development Helper Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in a devcontainer
check_devcontainer() {
    if [[ -z "${REMOTE_CONTAINERS}" ]]; then
        echo_warn "This script is designed to run in a VS Code devcontainer"
        echo_warn "Some features may not work as expected"
    else
        echo_info "Running in devcontainer environment ✓"
    fi
}

# Install/update dependencies
install_deps() {
    echo_info "Installing dependencies with UV..."
    uv sync --frozen
    echo_info "Dependencies installed ✓"
}

# Run tests
run_tests() {
    echo_info "Running tests..."
    python -m pytest tests/ -v
}

# Run specific test file
run_test_file() {
    if [[ -z "$1" ]]; then
        echo_error "Please specify a test file"
        echo "Usage: $0 test <test_file>"
        exit 1
    fi
    echo_info "Running test file: $1"
    python -m pytest "tests/$1" -v
}

# Check code quality
check_quality() {
    echo_info "Running code quality checks..."
    
    # Check if ruff is available
    if command -v ruff &> /dev/null; then
        echo_info "Running ruff linter..."
        ruff check cs336_basics/
        echo_info "Running ruff formatter check..."
        ruff format --check cs336_basics/
    else
        echo_warn "Ruff not found, skipping linting"
    fi
}

# Format code
format_code() {
    echo_info "Formatting code..."
    if command -v ruff &> /dev/null; then
        ruff format cs336_basics/
        echo_info "Code formatted ✓"
    else
        echo_error "Ruff not found"
        exit 1
    fi
}

# Show help
show_help() {
    echo "CS336 Assignment 1 - Development Helper"
    echo ""
    echo "Usage: $0 <command> [args]"
    echo ""
    echo "Commands:"
    echo "  install         Install/update dependencies"
    echo "  test           Run all tests"
    echo "  test <file>    Run specific test file"
    echo "  check          Run code quality checks"
    echo "  format         Format code"
    echo "  help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 install"
    echo "  $0 test"
    echo "  $0 test test_tokenizer.py"
    echo "  $0 check"
    echo "  $0 format"
}

# Main script logic
main() {
    check_devcontainer
    
    case "${1:-help}" in
        "install")
            install_deps
            ;;
        "test")
            if [[ -n "$2" ]]; then
                run_test_file "$2"
            else
                run_tests
            fi
            ;;
        "check")
            check_quality
            ;;
        "format")
            format_code
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

main "$@"
