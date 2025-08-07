#!/usr/bin/env python3
"""
Poly Haven Texture Downloader for EzSim

This script downloads indoor textures from Poly Haven's free texture library,
processes them to match the format used by EzSim (similar to checker.png),
and saves them in the textures folder with systematic naming.

Features:
- Downloads diffuse textures only (PNG format)
- Downloads 1k resolution (1024x1024) for high quality
- Converts to RGB format for consistency with EzSim
- Systematic naming convention: indoor_texture_name.png
- Filters for indoor/architectural textures
- Skips existing files to avoid re-downloading
"""

import os
import json
import requests
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
import numpy as np
from urllib.parse import urljoin
import argparse


class PolyHavenTextureDownloader:
    """Download and process textures from Poly Haven API"""
    
    BASE_URL = "https://api.polyhaven.com"
    ASSETS_URL = "https://api.polyhaven.com/assets"
    FILES_URL = "https://api.polyhaven.com/files"
    
    def __init__(self, output_dir: str = "ezsim/assets/textures", 
                 target_resolution: int = 1024, 
                 max_downloads: int = 20):
        """
        Initialize the downloader
        
        Parameters:
        -----------
        output_dir : str
            Directory to save downloaded textures
        target_resolution : int
            Target resolution for downloaded textures (默认1024，保持1k纹理原始质量)
        max_downloads : int
            Maximum number of textures to download
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_resolution = target_resolution
        self.max_downloads = max_downloads
        
        # Indoor/architectural categories we're interested in
        self.indoor_keywords = [
            'wood', 'tile', 'fabric', 'carpet', 'brick', 'concrete', 
            'marble', 'stone', 'leather', 'metal', 'floor', 'wall',
            'ceramic', 'plaster', 'paint', 'glass', 'plastic',
            'laminate', 'veneer', 'panel', 'board', 'parquet',
            'hardwood', 'oak', 'pine', 'mahogany', 'walnut', 'bamboo',
            'granite', 'limestone', 'slate', 'travertine', 'sandstone',
            'vinyl', 'linoleum', 'cork', 'rubber', 'textile'
        ]
        
        # Indoor categories to prioritize
        self.indoor_categories = [
            'indoor', 'architectural', 'fabric', 'wood', 'metal', 
            'concrete', 'stone', 'tiles', 'flooring', 'wall'
        ]
        
        # Map texture types to file suffixes (只需要diffuse纹理)
        self.texture_types = {
            'Diffuse': '_diffuse',
            # 'nor_gl': '_normal',
            # 'Rough': '_roughness',
            # 'Displacement': '_displacement',
            # 'AO': '_ao'
        }
        
        print(f"✓ Initialized downloader")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Target resolution: Keep original (1k=1024x1024)")
        print(f"  Max downloads: {self.max_downloads}")
        print(f"  Format: PNG diffuse textures only")
        
    def get_texture_assets(self) -> Dict:
        """Get list of texture assets from Poly Haven API"""
        print("\n📡 Fetching texture assets from Poly Haven...")
        
        try:
            # First try to get indoor category specifically
            response = requests.get(f"{self.ASSETS_URL}?t=textures&c=indoor", timeout=10)
            response.raise_for_status()
            indoor_assets = response.json()
            
            if indoor_assets:
                print(f"✓ Found {len(indoor_assets)} indoor texture assets")
                return indoor_assets
            else:
                print("⚠️ No indoor category assets found, trying all textures...")
                
        except requests.RequestException as e:
            print(f"⚠️ Indoor category request failed: {e}")
            print("   Falling back to all textures...")
        
        try:
            # Fallback to all textures
            response = requests.get(f"{self.ASSETS_URL}?t=textures", timeout=10)
            response.raise_for_status()
            assets = response.json()
            
            print(f"✓ Found {len(assets)} total texture assets")
            return assets
            
        except requests.RequestException as e:
            print(f"❌ Error fetching assets: {e}")
            return {}
    
    def filter_indoor_textures(self, assets: Dict) -> List[Tuple[str, Dict]]:
        """Filter assets for indoor/architectural textures"""
        print("\n🏠 Filtering for indoor textures...")
        
        indoor_textures = []
        
        for asset_id, asset_info in assets.items():
            name = asset_info.get('name', '').lower()
            tags = [tag.lower() for tag in asset_info.get('tags', [])]
            categories = [cat.lower() for cat in asset_info.get('categories', [])]
            
            # Check if this is an indoor/architectural texture
            is_indoor = False
            indoor_score = 0  # Score to prioritize better matches
            
            # High priority: indoor category match
            if any(cat in self.indoor_categories for cat in categories):
                is_indoor = True
                indoor_score += 10
            
            # Medium priority: name contains indoor keywords
            name_matches = sum(1 for keyword in self.indoor_keywords if keyword in name)
            if name_matches > 0:
                is_indoor = True
                indoor_score += name_matches * 3
            
            # Low priority: tags contain indoor keywords
            tag_matches = sum(1 for tag in tags for keyword in self.indoor_keywords if keyword in tag)
            if tag_matches > 0:
                is_indoor = True
                indoor_score += tag_matches
            
            # Exclude outdoor/nature textures
            outdoor_keywords = ['aerial', 'ground', 'sand', 'rock', 'cliff', 'mountain', 
                              'grass', 'dirt', 'soil', 'beach', 'snow', 'ice']
            has_outdoor = any(keyword in name or keyword in ' '.join(tags) 
                            for keyword in outdoor_keywords)
            
            if is_indoor and not has_outdoor:
                indoor_textures.append((asset_id, asset_info, indoor_score))
        
        print(f"✓ Found {len(indoor_textures)} indoor textures")
        
        # Sort by indoor score (descending) then by name
        indoor_textures.sort(key=lambda x: (-x[2], x[1].get('name', '')))
        
        # Return top matches without score
        result = [(asset_id, asset_info) for asset_id, asset_info, score in indoor_textures[:self.max_downloads]]
        
        # Show what we found
        if result:
            print("📋 Selected textures:")
            for i, (asset_id, asset_info) in enumerate(result[:5], 1):
                name = asset_info.get('name', asset_id)
                categories = asset_info.get('categories', [])
                print(f"   {i}. {name} (categories: {categories})")
            if len(result) > 5:
                print(f"   ... and {len(result) - 5} more")
        
        return result
    
    def get_asset_files(self, asset_id: str) -> Optional[Dict]:
        """Get file information for a specific asset"""
        try:
            response = requests.get(f"{self.FILES_URL}/{asset_id}", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"  ❌ Error fetching files for {asset_id}: {e}")
            return None
    
    def download_and_process_texture(self, asset_id: str, asset_info: Dict) -> bool:
        """Download and process a single texture asset"""
        name = asset_info.get('name', asset_id)
        print(f"\n📥 Processing: {name}")
        
        # Get file information
        files_info = self.get_asset_files(asset_id)
        if not files_info:
            return False
        
        # Download each texture type we're interested in
        success_count = 0
        for tex_type, suffix in self.texture_types.items():
            if tex_type in files_info:
                # Get available resolutions for this texture type
                tex_resolutions = files_info[tex_type]
                if not tex_resolutions:
                    continue
                
                # 获取可用分辨率并优先选择1k分辨率获得更高质量的纹理
                available_resolutions = list(tex_resolutions.keys())
                
                chosen_res_key = None
                if '1k' in available_resolutions:
                    chosen_res_key = '1k'
                elif '2k' in available_resolutions:
                    chosen_res_key = '2k'
                elif available_resolutions:
                    chosen_res_key = available_resolutions[0]
                
                if chosen_res_key and chosen_res_key in tex_resolutions:
                    # Get format info (只使用PNG格式)
                    format_options = tex_resolutions[chosen_res_key]
                    
                    if 'png' in format_options:
                        file_info = format_options['png']
                        print(f"  📐 Using {tex_type} at {chosen_res_key} resolution (png)")
                        if self._download_texture_file(asset_id, name, tex_type, suffix, file_info):
                            success_count += 1
                    else:
                        print(f"  ⚠️  PNG format not available for {tex_type}")
        
        return success_count > 0
    
    def _download_texture_file(self, asset_id: str, asset_name: str, 
                             tex_type: str, suffix: str, file_info: Dict) -> bool:
        """Download and process a single texture file"""
        
        # Generate filename (简化命名，去掉_diffuse后缀)
        safe_name = self._sanitize_filename(asset_name)
        filename = f"indoor_{safe_name}.png"
        filepath = self.output_dir / filename
        
        # Skip if already exists
        if filepath.exists():
            print(f"  ⏭️  Skipping {tex_type}: already exists")
            return True
        
        # Get download URL
        download_url = file_info.get('url')
        if not download_url:
            print(f"  ❌ No download URL for {tex_type}")
            return False
        
        try:
            print(f"  ⬇️  Downloading {tex_type}...")
            
            # Download file
            response = requests.get(download_url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Load and process image (PNG already loaded, just need to convert format if needed)
            image = Image.open(response.raw)
            
            # Convert to RGB if needed (remove alpha channel)
            if image.mode == 'RGBA':
                # Create white background for transparency
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 保持原始分辨率，不进行缩放（1k通常是1024x1024）
            # 只有当图像不是正方形时才调整尺寸
            if image.size[0] != image.size[1]:
                # 如果不是正方形，裁剪为正方形（使用较小的边）
                min_size = min(image.size)
                left = (image.size[0] - min_size) // 2
                top = (image.size[1] - min_size) // 2
                right = left + min_size
                bottom = top + min_size
                image = image.crop((left, top, right, bottom))
                print(f"  📐 Cropped to square: {image.size}")
            else:
                print(f"  📐 Keeping original size: {image.size}")
            
            # Save as PNG
            image.save(filepath, 'PNG', optimize=True)
            
            # Verify saved image
            if filepath.exists() and filepath.stat().st_size > 0:
                print(f"  ✅ Saved: {filename}")
                return True
            else:
                print(f"  ❌ Failed to save: {filename}")
                return False
                
        except Exception as e:
            print(f"  ❌ Error downloading {tex_type}: {e}")
            return False
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize asset name for use in filename"""
        # Convert to lowercase and replace spaces/special chars with underscores
        safe_name = name.lower()
        safe_name = ''.join(c if c.isalnum() else '_' for c in safe_name)
        # Remove multiple consecutive underscores
        while '__' in safe_name:
            safe_name = safe_name.replace('__', '_')
        # Remove leading/trailing underscores
        safe_name = safe_name.strip('_')
        return safe_name
    
    def run(self):
        """Main execution function"""
        print("🎨 EzSim Texture Downloader - Poly Haven Indoor Textures")
        print("=" * 60)
        
        # Check existing textures
        existing_textures = list(self.output_dir.glob("indoor_*.png"))
        print(f"📁 Found {len(existing_textures)} existing indoor textures")
        
        # Get all texture assets
        assets = self.get_texture_assets()
        if not assets:
            print("❌ No assets found, exiting")
            return
        
        # Filter for indoor textures
        indoor_textures = self.filter_indoor_textures(assets)
        if not indoor_textures:
            print("❌ No indoor textures found, exiting")
            return
        
        # Download and process textures
        print(f"\n🚀 Starting download of {len(indoor_textures)} textures...")
        print("-" * 60)
        
        success_count = 0
        for i, (asset_id, asset_info) in enumerate(indoor_textures, 1):
            print(f"\n[{i}/{len(indoor_textures)}]", end="")
            
            if self.download_and_process_texture(asset_id, asset_info):
                success_count += 1
            
            # Rate limiting
            time.sleep(0.5)
        
        # Summary
        print("\n" + "=" * 60)
        print(f"✅ Download completed!")
        print(f"   Successfully processed: {success_count}/{len(indoor_textures)} textures")
        
        # List downloaded files
        final_textures = list(self.output_dir.glob("indoor_*.png"))
        print(f"   Total indoor textures: {len(final_textures)}")
        
        if final_textures:
            print(f"\n📋 Downloaded textures saved in: {self.output_dir}")
            for texture in sorted(final_textures)[:10]:  # Show first 10
                print(f"   - {texture.name}")
            if len(final_textures) > 10:
                print(f"   ... and {len(final_textures) - 10} more")


def check_existing_texture_format():
    """Check the format of existing checker.png texture"""
    checker_path = Path("ezsim/assets/textures/checker.png")
    
    if checker_path.exists():
        try:
            with Image.open(checker_path) as img:
                print(f"📋 Existing texture format (checker.png):")
                print(f"   Size: {img.size}")
                print(f"   Mode: {img.mode}")
                print(f"   Format: {img.format}")
                return img.size[0]  # Return width as target resolution
        except Exception as e:
            print(f"❌ Error reading checker.png: {e}")
    
    return 512  # Default resolution


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Download indoor textures from Poly Haven")
    parser.add_argument("--resolution", type=int, default=None, 
                       help="Target resolution (default: match checker.png)")
    parser.add_argument("--max-downloads", type=int, default=20,
                       help="Maximum number of textures to download")
    parser.add_argument("--output-dir", type=str, default="ezsim/assets/textures",
                       help="Output directory for textures")
    
    args = parser.parse_args()
    
    # Check existing texture format if no resolution specified
    if args.resolution is None:
        args.resolution = check_existing_texture_format()
        print(f"🎯 Using resolution: {args.resolution}x{args.resolution}")
    
    # Create and run downloader
    downloader = PolyHavenTextureDownloader(
        output_dir=args.output_dir,
        target_resolution=args.resolution,
        max_downloads=args.max_downloads
    )
    
    try:
        downloader.run()
    except KeyboardInterrupt:
        print("\n\n⛔ Download interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
