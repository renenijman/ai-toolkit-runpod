#!/usr/bin/env python3
"""
Model download and verification script for HiDream training
This script downloads the required models and verifies they work offline
"""

import os
import sys
import time
from huggingface_hub import snapshot_download
from pathlib import Path

def download_with_retry(repo_id, max_retries=3):
    """Download a model with retry logic"""
    for attempt in range(max_retries):
        try:
            print(f'📥 Downloading {repo_id} (attempt {attempt + 1}/{max_retries})...')
            snapshot_download(
                repo_id=repo_id,
                cache_dir='/opt/huggingface_cache',
                resume_download=True,
                local_files_only=False
            )
            print(f'✅ {repo_id} downloaded successfully!')
            return True
        except Exception as e:
            print(f'⚠️ Attempt {attempt + 1} failed for {repo_id}: {e}')
            if attempt < max_retries - 1:
                time.sleep(10)
            else:
                print(f'❌ FATAL: Failed to download {repo_id} after {max_retries} attempts')
                sys.exit(1)  # HARD FAIL THE BUILD
    return False

def verify_models():
    """Verify that models can be loaded offline"""
    print('🎉 All models downloaded - now verifying...')
    
    # STRICT VERIFICATION - Build fails if ANY of this fails
    try:
        from transformers import AutoTokenizer, AutoConfig
        
        # Test 1: Llama model offline loading
        print('🔍 Test 1: Loading Llama tokenizer offline...')
        tokenizer = AutoTokenizer.from_pretrained(
            'unsloth/Meta-Llama-3.1-8B-Instruct',
            cache_dir='/opt/huggingface_cache',
            local_files_only=True  # MUST work offline
        )
        print('✅ Test 1 PASSED')
        
        # Test 2: Llama config offline loading  
        print('🔍 Test 2: Loading Llama config offline...')
        llama_config = AutoConfig.from_pretrained(
            'unsloth/Meta-Llama-3.1-8B-Instruct',
            cache_dir='/opt/huggingface_cache', 
            local_files_only=True  # MUST work offline
        )
        print('✅ Test 2 PASSED')
        
        # Test 3: HiDream config offline loading
        print('🔍 Test 3: Loading HiDream config offline...')
        hidream_config = AutoConfig.from_pretrained(
            'HiDream-ai/HiDream-I1-Full',
            cache_dir='/opt/huggingface_cache',
            local_files_only=True  # MUST work offline
        )
        print('✅ Test 3 PASSED')
        
        # Test 4: Check cache size
        cache_path = Path('/opt/huggingface_cache')
        if cache_path.exists():
            total_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
            size_gb = total_size / (1024**3)
            print(f'📊 Cache size: {size_gb:.1f} GB')
            
            if size_gb < 25:  # Should be at least 25GB for both models
                print(f'❌ FATAL: Cache too small ({size_gb:.1f} GB), models not fully cached')
                sys.exit(1)
        else:
            print('❌ FATAL: Cache directory does not exist')
            sys.exit(1)
        
        print('🎉 ALL VERIFICATION TESTS PASSED!')
        print('💡 Training will start instantly without downloads')
        
    except Exception as e:
        print(f'❌ FATAL VERIFICATION FAILURE: {e}')
        print('🚨 BUILD MUST FAIL - models not properly cached')
        sys.exit(1)  # HARD FAIL THE BUILD

def main():
    """Main function to download and verify models"""
    print('🚀 Starting model pre-download process...')
    
    # Download both models (exactly matching your config)
    download_with_retry('HiDream-ai/HiDream-I1-Full')
    download_with_retry('unsloth/Meta-Llama-3.1-8B-Instruct')
    
    # Verify everything works
    verify_models()
    
    print('🎊 Model caching complete and verified!')

if __name__ == '__main__':
    main()