"""
Vision-Language Model Benchmark using Local Dataset
==================================================

Benchmarks vision-language models for image captioning using:
- Hugging Face InferenceClient 
- GitHub Models API (GPT-4.1)
- Local image dataset for realistic evaluation

Tests multiple providers and models for latency and caption quality.
"""

import os
import time
import json
import statistics
import pandas as pd
import requests
import base64
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from huggingface_hub import InferenceClient
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import glob
from pathlib import Path

class VisionLanguageBenchmark:
    def __init__(self, hf_token=None, github_token=None, dataset_path=None):
        # Tokens
        self.hf_token = hf_token or "your_key"
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN", "your_key")
        
        # Dataset path - modify this to point to your local dataset
        self.dataset_path = dataset_path or "./dataset"  # Change this to your dataset path
        
        # Models to test with their providers
        self.models = {
            "qwen_2_5_vl": {
                "model": "Qwen/Qwen2.5-VL-7B-Instruct", 
                "provider": "hyperbolic",
                "type": "huggingface",
                "description": "Qwen 2.5 VL - Strong image-language reasoning"
            },
            "llama_4_scout": {
                "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                "provider": "novita", 
                "type": "huggingface",
                "description": "Llama 4 Scout - Balanced language + vision"
            },
            "gemma_3_27b": {
                "model": "google/gemma-3-27b-it",
                "provider": "nebius",
                "type": "huggingface",
                "description": "Gemma 3 27B - Large instruction-tuned model with vision"
            },
            "github_gpt41": {
                "model": "openai/gpt-4.1",
                "provider": "github",
                "type": "github",
                "description": "GitHub Models GPT-4.1 - Advanced vision-language model"
            }
        }
        
        self.results = {model_id: [] for model_id in self.models.keys()}
        self.clients = {}
        
        # Initialize clients for each provider
        self.init_clients()
    
    def init_clients(self):
        """Initialize clients for each provider"""
        # Initialize HuggingFace clients
        hf_providers = set(model["provider"] for model in self.models.values() if model["type"] == "huggingface")
        
        for provider in hf_providers:
            try:
                self.clients[provider] = InferenceClient(
                    provider=provider,
                    api_key=self.hf_token
                )
                print(f"✅ Initialized HF client for provider: {provider}")
            except Exception as e:
                print(f"❌ Failed to initialize HF {provider}: {e}")
        
        # Initialize GitHub Models client
        try:
            self.clients["github"] = ChatCompletionsClient(
                endpoint="https://models.github.ai/inference",
                credential=AzureKeyCredential(self.github_token),
            )
            print(f"✅ Initialized GitHub Models client")
        except Exception as e:
            print(f"❌ Failed to initialize GitHub Models client: {e}")
    
    def load_local_dataset(self, max_images=None):
        """Load images from local dataset"""
        print(f"📂 Loading images from: {self.dataset_path}")
        
        # Check if dataset path exists
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        
        # Supported image extensions
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.tiff", "*.webp"]
        
        # Find all image files
        image_paths = []
        for ext in image_extensions:
            # Search recursively for images
            pattern = os.path.join(self.dataset_path, "**", ext)
            image_paths.extend(glob.glob(pattern, recursive=True))
        
        # Sort for consistent ordering
        image_paths.sort()
        
        if not image_paths:
            raise ValueError(f"No images found in {self.dataset_path}")
        
        # Limit number of images if specified
        if max_images:
            image_paths = image_paths[:max_images]
        
        print(f"✅ Found {len(image_paths)} images")
        
        # Validate images
        valid_images = []
        for img_path in tqdm(image_paths, desc="Validating images"):
            try:
                with Image.open(img_path) as img:
                    # Basic validation - ensure image can be opened
                    img.verify()
                valid_images.append(img_path)
            except Exception as e:
                print(f"⚠️ Skipping invalid image {img_path}: {e}")
        
        print(f"✅ Validated {len(valid_images)} images")
        return valid_images
    
    def get_image_info(self, image_path):
        """Get basic information about an image"""
        try:
            with Image.open(image_path) as img:
                return {
                    "filename": os.path.basename(image_path),
                    "size": img.size,
                    "mode": img.mode,
                    "format": img.format,
                    "file_size": os.path.getsize(image_path)
                }
        except Exception as e:
            return {
                "filename": os.path.basename(image_path),
                "error": str(e)
            }
    
    def encode_image_to_base64(self, image_path):
        """Convert image to base64 for GitHub Models API"""
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
            return img_data
    
    def upload_image_to_temp_url(self, image_path):
        """
        Convert local image to a publicly accessible URL for HuggingFace
        For this demo, we'll use base64 data URLs
        """
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
            return f"data:image/jpeg;base64,{img_data}"
    
    def generate_caption_huggingface(self, model_id, image_path, max_retries=3):
        """Generate caption using HuggingFace InferenceClient"""
        model_config = self.models[model_id]
        provider = model_config["provider"]
        model_name = model_config["model"]
        
        if provider not in self.clients:
            return f"Provider {provider} not available", 0, False
        
        client = self.clients[provider]
        
        # Convert image to URL
        image_url = self.upload_image_to_temp_url(image_path)
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this image in one detailed, descriptive sentence."},
                                {"type": "image_url", "image_url": {"url": image_url}}
                            ]
                        }
                    ],
                    max_tokens=150,
                    temperature=0.3
                )
                
                end_time = time.time()
                latency = end_time - start_time
                
                caption = completion.choices[0].message.content
                return caption, latency, True
                
            except Exception as e:
                error_msg = str(e)
                if "rate limit" in error_msg.lower() and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif attempt < max_retries - 1:
                    print(f"Error with {model_id}, retrying: {error_msg}")
                    time.sleep(2)
                    continue
                else:
                    return f"Error: {error_msg}", 0, False
        
        return "Max retries exceeded", 0, False
    
    def generate_caption_github(self, model_id, image_path, max_retries=3):
        """Generate caption using GitHub Models API"""
        model_config = self.models[model_id]
        model_name = model_config["model"]
        
        if "github" not in self.clients:
            return "GitHub client not available", 0, False
        
        client = self.clients["github"]
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                # Convert image to base64
                base64_image = self.encode_image_to_base64(image_path)
                
                # Create system message
                system_message = SystemMessage(
                    "You are an expert image analyst. Describe the image in one detailed, descriptive sentence."
                )
                
                # Build user content with text and image
                user_content = [
                    {"type": "text", "text": "Describe this image in one detailed, descriptive sentence."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
                
                # Make API call
                response = client.complete(
                    messages=[system_message, UserMessage(content=user_content)],
                    temperature=0.3,
                    top_p=0.9,
                    model=model_name,
                    max_tokens=150
                )
                
                end_time = time.time()
                latency = end_time - start_time
                
                caption = response.choices[0].message.content
                return caption, latency, True
                
            except Exception as e:
                error_msg = str(e)
                if "rate limit" in error_msg.lower() and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif attempt < max_retries - 1:
                    print(f"Error with {model_id}, retrying: {error_msg}")
                    time.sleep(2)
                    continue
                else:
                    return f"Error: {error_msg}", 0, False
        
        return "Max retries exceeded", 0, False
    
    def generate_caption(self, model_id, image_path, max_retries=3):
        """Generate caption using appropriate client based on model type"""
        model_config = self.models[model_id]
        
        if model_config["type"] == "huggingface":
            return self.generate_caption_huggingface(model_id, image_path, max_retries)
        elif model_config["type"] == "github":
            return self.generate_caption_github(model_id, image_path, max_retries)
        else:
            return f"Unknown model type: {model_config['type']}", 0, False
    
    def benchmark_image(self, image_path, image_idx):
        """Test all models on one image"""
        image_name = os.path.basename(image_path)
        image_info = self.get_image_info(image_path)
        
        print(f"\nProcessing image {image_idx + 1}: {image_name}")
        print(f"  Size: {image_info.get('size', 'unknown')}, Format: {image_info.get('format', 'unknown')}")
        
        for model_id in self.models.keys():
            print(f"  Testing {model_id}...", end=" ")
            
            caption, latency, success = self.generate_caption(model_id, image_path)
            
            self.results[model_id].append({
                "image_idx": image_idx,
                "image_path": image_path,
                "image_name": image_name,
                "image_info": image_info,
                "caption": caption,
                "latency": latency,
                "success": success,
                "timestamp": datetime.now().isoformat()
            })
            
            status = "✅" if success else "❌"
            print(f"{status} {latency:.2f}s")
            print(f"    Caption: {caption[:100]}{'...' if len(caption) > 100 else ''}")
            
            # Delay between requests to avoid rate limiting
            time.sleep(2)
    
    def run_benchmark(self, max_images=10):
        """Run the complete benchmark"""
        print(f"🚀 Starting Vision-Language Model Benchmark with Local Dataset")
        print(f"🔧 Testing {len(self.models)} models on up to {max_images} images")
        print(f"🏷️ Models: {', '.join(self.models.keys())}")
        
        # Load images from local dataset
        try:
            image_paths = self.load_local_dataset(max_images)
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return {}
        
        if not image_paths:
            print("❌ No valid images found in dataset")
            return {}
        
        print(f"\n🖼️ Processing {len(image_paths)} images...")
        
        # Test each image
        for idx, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
            self.benchmark_image(image_path, idx)
        
        print("\n✅ Benchmark completed!")
        return self.analyze_results()
    
    def analyze_results(self):
        """Analyze benchmark results"""
        print("📊 Analyzing results...")
        
        analysis = {}
        
        for model_id, results in self.results.items():
            model_config = self.models[model_id]
            
            if results:
                successful = [r for r in results if r["success"]]
                failed = [r for r in results if not r["success"]]
                
                if successful:
                    latencies = [r["latency"] for r in successful]
                    caption_lengths = [len(r["caption"]) for r in successful]
                    
                    analysis[model_id] = {
                        "model_name": model_config["model"],
                        "provider": model_config["provider"],
                        "type": model_config["type"],
                        "description": model_config["description"],
                        "total_requests": len(results),
                        "successful": len(successful),
                        "failed": len(failed),
                        "success_rate": len(successful) / len(results) * 100,
                        "avg_latency": statistics.mean(latencies),
                        "median_latency": statistics.median(latencies),
                        "min_latency": min(latencies),
                        "max_latency": max(latencies),
                        "std_latency": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                        "avg_caption_length": statistics.mean(caption_lengths),
                        "sample_captions": [r["caption"] for r in successful[:3]],
                        "sample_images": [r["image_name"] for r in successful[:3]]
                    }
                else:
                    analysis[model_id] = {
                        "model_name": model_config["model"],
                        "provider": model_config["provider"], 
                        "type": model_config["type"],
                        "description": model_config["description"],
                        "total_requests": len(results),
                        "successful": 0,
                        "failed": len(failed),
                        "success_rate": 0,
                        "avg_latency": 0,
                        "median_latency": 0,
                        "min_latency": 0,
                        "max_latency": 0,
                        "std_latency": 0,
                        "avg_caption_length": 0,
                        "sample_captions": [],
                        "sample_images": []
                    }
        
        return analysis
    
    def create_report(self, analysis):
        """Create comprehensive report"""
        print("\n" + "="*80)
        print("📊 VISION-LANGUAGE MODEL BENCHMARK RESULTS")
        print("="*80)
        
        # Summary table
        data = []
        for model_id, stats in analysis.items():
            data.append({
                "Model ID": model_id,
                "Type": stats["type"],
                "Provider": stats["provider"],
                "Success Rate": f"{stats['success_rate']:.1f}%",
                "Avg Latency": f"{stats['avg_latency']:.2f}s",
                "Median Latency": f"{stats['median_latency']:.2f}s",
                "Min/Max": f"{stats['min_latency']:.2f}s / {stats['max_latency']:.2f}s",
                "Avg Caption Length": f"{stats['avg_caption_length']:.0f} chars",
                "Requests": f"{stats['successful']}/{stats['total_requests']}"
            })
        
        df = pd.DataFrame(data)
        print("\n📈 PERFORMANCE SUMMARY:")
        print(df.to_string(index=False))
        
        # Provider comparison
        print("\n🏢 PROVIDER COMPARISON:")
        provider_stats = {}
        for model_id, stats in analysis.items():
            provider = stats["provider"]
            if provider not in provider_stats:
                provider_stats[provider] = {
                    "models": 0,
                    "avg_success_rate": [],
                    "avg_latency": []
                }
            provider_stats[provider]["models"] += 1
            provider_stats[provider]["avg_success_rate"].append(stats["success_rate"])
            if stats["avg_latency"] > 0:
                provider_stats[provider]["avg_latency"].append(stats["avg_latency"])
        
        for provider, stats in provider_stats.items():
            avg_success = statistics.mean(stats["avg_success_rate"]) if stats["avg_success_rate"] else 0
            avg_latency = statistics.mean(stats["avg_latency"]) if stats["avg_latency"] else 0
            print(f"  {provider}: {stats['models']} models, {avg_success:.1f}% success rate, {avg_latency:.2f}s avg latency")
        
        # Top performers
        successful_models = {k: v for k, v in analysis.items() if v['success_rate'] > 0}
        
        if successful_models:
            print("\n🏆 TOP PERFORMERS:")
            
            fastest = min(successful_models.items(), key=lambda x: x[1]['avg_latency'])
            most_reliable = max(successful_models.items(), key=lambda x: x[1]['success_rate'])
            
            print(f"⚡ Fastest Model: {fastest[0]} ({fastest[1]['avg_latency']:.2f}s avg) - {fastest[1]['provider']}")
            print(f"🔒 Most Reliable: {most_reliable[0]} ({most_reliable[1]['success_rate']:.1f}% success) - {most_reliable[1]['provider']}")
            
            # Sample captions comparison
            print(f"\n📝 SAMPLE CAPTIONS COMPARISON:")
            for model_id, stats in list(successful_models.items())[:2]:  # Show top 2 models
                print(f"\n{model_id} ({stats['provider']}):")
                for i, (caption, image) in enumerate(zip(stats['sample_captions'][:2], stats['sample_images'][:2])):
                    print(f"  Image: {image}")
                    print(f"  Caption: {caption}")
                    print()
        
        print("="*80)
        return df
    
    def save_results(self, analysis, filename="vl_benchmark_results.json"):
        """Save detailed results"""
        save_data = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "benchmark_type": "vision_language_local_dataset",
                "dataset_path": self.dataset_path,
                "models_tested": len(self.models),
                "total_images": len(self.results[list(self.results.keys())[0]]) if self.results else 0,
                "providers": list(set(model["provider"] for model in self.models.values()))
            },
            "models": self.models,
            "analysis": analysis,
            "detailed_results": self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"💾 Detailed results saved to {filename}")

def main():
    """Main benchmark function"""
    print("🎯 Vision-Language Model Benchmark - Local Dataset Evaluation")
    print("Testing HuggingFace InferenceClient + GitHub Models API for image captioning\n")
    
    # Configuration - MODIFY THESE PATHS FOR YOUR SETUP
    DATASET_PATH = "./dataset"  # Change this to your dataset path
    MAX_IMAGES = 20             # Number of images to test (set to None for all images)
    
    print(f"📂 Dataset path: {DATASET_PATH}")
    print(f"🔢 Max images: {MAX_IMAGES}")
    
    # Initialize benchmark
    benchmark = VisionLanguageBenchmark(dataset_path=DATASET_PATH)
    
    try:
        # Run benchmark
        analysis = benchmark.run_benchmark(max_images=MAX_IMAGES)
        
        if not analysis:
            print("❌ No analysis results - benchmark may have failed")
            return
        
        # Create and display report
        df = benchmark.create_report(analysis)
        
        # Save results
        benchmark.save_results(analysis)
        df.to_csv("vl_benchmark_summary.csv", index=False)
        
        print("\n✅ Benchmark completed successfully!")
        print("📁 Files created:")
        print("  - vl_benchmark_results.json (detailed results)")
        print("  - vl_benchmark_summary.csv (summary table)")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()