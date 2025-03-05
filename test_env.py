#!/usr/bin/env python
"""
Test script to verify that environment variables are loading correctly.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check for Azure OpenAI environment variables
azure_openai_key = os.environ.get("AZURE_OPENAI_API_KEY")
azure_openai_version = os.environ.get("AZURE_OPENAI_API_VERSION")
azure_openai_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")

print("Azure OpenAI Environment Variables:")
print(f"  AZURE_OPENAI_API_KEY: {'✅ Found' if azure_openai_key else '❌ Not found'}")
print(f"  AZURE_OPENAI_API_VERSION: {'✅ Found' if azure_openai_version else '❌ Not found'}")
print(f"  AZURE_OPENAI_DEPLOYMENT_NAME: {'✅ Found' if azure_openai_deployment else '❌ Not found'}")
print(f"  AZURE_OPENAI_ENDPOINT: {'✅ Found' if azure_openai_endpoint else '❌ Not found'}")

# Check for OpenAI environment variable
openai_key = os.environ.get("OPENAI_API_KEY")
print("\nOpenAI Environment Variable:")
print(f"  OPENAI_API_KEY: {'✅ Found' if openai_key else '❌ Not found'}")

# Show first few characters of keys (for security)
if azure_openai_key:
    print(f"\nAZURE_OPENAI_API_KEY starts with: {azure_openai_key[:5]}...")
if azure_openai_version:
    print(f"AZURE_OPENAI_API_VERSION: {azure_openai_version}")
if azure_openai_deployment:
    print(f"AZURE_OPENAI_DEPLOYMENT_NAME: {azure_openai_deployment}")
if azure_openai_endpoint:
    print(f"AZURE_OPENAI_ENDPOINT: {azure_openai_endpoint}") 