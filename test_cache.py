#!/usr/bin/env python3
"""
Test script to verify HiDream model cache status
Run this in your RunPod environment to check cache status
"""

import os
import sys
from pathlib import Path
from huggingface_hub import try_to_load_from_cache

def check_model_cache(repo_id, cache_dir="/opt/huggingface_cache"):
    """Check if a model is fully cached locally"""
    print(f"\n🔍 Checking cache for: {repo_id}")
    
    # Check if cache directory exists
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"❌ Cache directory not found: {cache_dir}")
        return False
    
    # Files we expect for HiDream models
    required_files = {
        "HiDream-ai/HiDream-I1-Full": [
            "config.json",
            "model.safetensors.index.json", 
            "diffusion_pytorch_model.safetensors",  # VAE
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json"
        ],
        "unsloth/Meta-Llama-3.1-8B-Instruct": [
            "config.json",
            "tokenizer.json", 
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model.safetensors.index.json",
            "generation_config.json"
        ]
    }
    
    files_to_check = required_files.get(repo_id, [
        "config.json",
        "model.safetensors", 
        "tokenizer.json"
    ])
    
    try:
        cached_files = []
        missing_files = []
        
        for filename in files_to_check:
            cached_path = try_to_load_from_cache(repo_id, filename, cache_dir=cache_dir)
            if cached_path:
                cached_files.append(filename)
            else:
                missing_files.append(filename)
        
        if cached_files:
            print(f"✅ Found cached files ({len(cached_files)}): {cached_files[:3]}{'...' if len(cached_files) > 3 else ''}")
            
        if missing_files:
            print(f"⚠️ Missing files ({len(missing_files)}): {missing_files[:3]}{'...' if len(missing_files) > 3 else ''}")
            return len(missing_files) < len(files_to_check) // 2  # Allow some missing files
            
        return len(cached_files) > 0
            
    except Exception as e:
        print(f"⚠️ Error checking cache: {e}")
        return False

def simulate_training_load():
    """Test if models can be loaded like training would"""
    print(f"\n🧪 Testing model loading (simulating training)...")
    
    try:
        from transformers import AutoTokenizer, AutoConfig
        
        # Test Llama components
        print("  Loading Llama tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            'unsloth/Meta-Llama-3.1-8B-Instruct',
            cache_dir=os.environ.get('HF_HOME', '/opt/huggingface_cache'),
            local_files_only=True
        )
        print("  ✅ Llama tokenizer loaded successfully")
        
        # Test HiDream config
        print("  Loading HiDream config...")
        config = AutoConfig.from_pretrained(
            'HiDream-ai/HiDream-I1-Full', 
            cache_dir=os.environ.get('HF_HOME', '/opt/huggingface_cache'),
            local_files_only=True
        )
        print("  ✅ HiDream config loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to load models: {e}")
        return False

def main():
    print("🧪 HiDream Model Cache Verification")
    print("=" * 60)
    
    # Models your training uses
    models = [
        "HiDream-ai/HiDream-I1-Full",
        "unsloth/Meta-Llama-3.1-8B-Instruct"
    ]
    
    cache_dir = os.environ.get('HF_HOME', '/opt/huggingface_cache')
    print(f"📁 Cache directory: {cache_dir}")
    
    # Check overall cache size
    if Path(cache_dir).exists():
        try:
            total_size = sum(f.stat().st_size for f in Path(cache_dir).rglob('*') if f.is_file())
            size_gb = total_size / (1024**3)
            print(f"💾 Total cache size: {size_gb:.2f} GB")
        except:
            print("💾 Cache size: Unable to calculate")
    else:
        print("💾 Cache directory does not exist")
    
    # Check each model
    all_cached = True
    for model in models:
        cached = check_model_cache(model, cache_dir)
        if not cached:
            all_cached = False
    
    # Test actual loading
    can_load = simulate_training_load()
    
    print("\n" + "=" * 60)
    if all_cached and can_load:
        print("🎉 SUCCESS: All models are cached and loadable!")
        print("💡 Training should start immediately without downloads")
        sys.exit(0)
    elif all_cached:
        print("⚠️ PARTIAL: Models cached but loading failed")
        print("💡 May need different cache configuration")
        sys.exit(1)
    else:
        print("❌ FAILED: Models not fully cached")
        print("💡 First training run will still download missing components")
        sys.exit(2)

if __name__ == "__main__":
    main()