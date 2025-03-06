# Using Azure OpenAI with Co-Scientist

This guide explains how to configure Co-Scientist to use Azure OpenAI API instead of standard OpenAI API.

## Problem

By default, Co-Scientist is configured to use the standard OpenAI API. If you want to use Azure OpenAI, there are some configuration issues that need to be addressed:

1. The application expects API credentials at the top level of the model configuration
2. However, the Azure OpenAI configuration in `config/azure_openai_config.yaml` has these credentials nested under `models.azure_openai`

## Solution

We've created a patched version of the model factory that properly handles Azure OpenAI configuration. This patch:

1. Extracts Azure OpenAI credentials from environment variables
2. Correctly passes these credentials to the Azure OpenAI model constructor

## Step-by-Step Usage Guide

1. Ensure your `.env` file contains the following Azure OpenAI environment variables:
   ```
   AZURE_OPENAI_API_KEY=your_api_key_here
   AZURE_OPENAI_API_VERSION=your_api_version_here (e.g., 2025-01-01-preview)
   AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name_here
   AZURE_OPENAI_ENDPOINT=your_endpoint_here (e.g., https://example.openai.azure.com)
   ```

2. Run the application using the patched script:
   ```
   .\run_azure.bat
   ```

## How the Patch Works

The patch modifies the `ModelFactory.create_model` method to:

1. Extract Azure OpenAI credentials from environment variables
2. Build a properly formatted configuration dictionary for the AzureOpenAIModel
3. Pass this configuration to the model constructor

This allows the Azure OpenAI model to be created with the correct credentials without modifying the original codebase.

## For Developers

If you want to integrate this patch into the main codebase, you would need to modify the `ModelFactory.create_model` method in `core/models/model_factory.py` to include the Azure OpenAI handling logic from our patch.

## Files

- `fixed_model_factory.py`: Contains the patched `ModelFactory` class with proper Azure OpenAI support
- `run_with_azure.py`: Entry point script that applies the patch and runs the application
- `run_azure.bat`: Batch file for Windows that runs the patched application

## Troubleshooting

If you encounter issues:

1. Verify your Azure OpenAI credentials are correct
2. Check that your Azure OpenAI deployment is active and available
3. Try the `use_azure_openai.py` script to test your Azure OpenAI connection directly
4. Look for error messages in the application logs 