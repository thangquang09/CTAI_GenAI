#!/bin/bash

# Parse command line arguments
DRIVE_LINK=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --drive_link)
            DRIVE_LINK="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --drive_link <google_drive_link>"
            exit 1
            ;;
    esac
done

# Check if drive_link is provided
if [ -z "$DRIVE_LINK" ]; then
    echo "Error: --drive_link argument is required"
    echo "Usage: $0 --drive_link <google_drive_link>"
    exit 1
fi

uv sync --quiet
gdown "$DRIVE_LINK" -O langchain_qdrant.zip
unzip -q langchain_qdrant.zip -d .
rm langchain_qdrant.zip

