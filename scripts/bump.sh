#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 New_Version_Number"
    exit 1
fi

NEW_VERSION_NUMBER=$1

find . -name "*.toml" | while read -r file; do
    sed -i '' -E "s/(version = )\"[0-9]+\.[0-9]+\.[0-9]+\"/\\1\"$NEW_VERSION_NUMBER\"/g" "$file"
done