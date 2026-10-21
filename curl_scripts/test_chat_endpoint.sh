#!/usr/bin/env bash
# Test POST /chat endpoint

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

curl -X POST ${BASE_URL}/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Give me 5 rules of SEI CERT C coding standard that I should follow for safe coding", "session_id": 1}'
