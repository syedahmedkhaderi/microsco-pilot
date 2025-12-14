"""SmartScan with Real AFM Data

Simple script to analyze real AFM h5 files with SmartScan.
"""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.h5_data_loader import DatasetManager, inspect_h5_structure
from src.vision_evaluator import VisionEvaluator
import numpy as np


def analyze_afm_file(h5_file_path: str, num_regions: int = 10):
    """Analyze a single AFM h5 file with SmartScan
    
    Args:
        h5_file_path: Path to AFM h5 file
        num_regions: Number of regions to extract and analyze
    """
    print("\n" + "="*70)
    print(f"  ANALYZING: {os.path.basename(h5_file_path)}")
    print("="*70)
    
    # Load AFM data
    with DatasetManager(h5_file_path) as dm:
        # Show metadata
        metadata = dm.get_metadata()
        print(f"\n📊 Dataset Information:")
        print(f"   AFM Mode: {metadata.get('afm_mode', 'unknown')}")
        print(f"   Resolution: {metadata.get('resolution', 'unknown')} px")
        print(f"   Scan Size: {metadata.get('scan_size', 'unknown')}")
        
        # Visualize raw data
        output_name = os.path.basename(h5_file_path).replace('.h5', '_visualization.png')
        output_path = f'results/afm_visualizations/{output_name}'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"\n🎨 Generating visualization...")
        dm.visualize_raw_data(save_path=output_path, cmap='afmhot')
        
        # Extract regions
        print(f"\n🔬 Extracting {num_regions} scan regions...")
        regions = dm.get_scan_regions(num_regions=num_regions)
        
        # Analyze with SmartScan
        print(f"\n🤖 Analyzing with SmartScan...")
        evaluator = VisionEvaluator()
        
        results = []
        for i, region in enumerate(regions):
            analysis = evaluator.analyze_region(region)
            results.append(analysis)
            print(f"   Region {i+1:2d}/{num_regions}: "
                  f"Quality={analysis['quality']:4.1f}/10, "
                  f"Complexity={analysis['complexity']:4.1f}/10, "
                  f"Sharpness={analysis['sharpness']:4.1f}/10")
        
        # Summary
        avg_quality = np.mean([r['quality'] for r in results])
        avg_complexity = np.mean([r['complexity'] for r in results])
        avg_sharpness = np.mean([r['sharpness'] for r in results])
        
        print(f"\n📈 Summary:")
        print(f"   Average Quality:    {avg_quality:.2f}/10")
        print(f"   Average Complexity: {avg_complexity:.2f}/10")
        print(f"   Average Sharpness:  {avg_sharpness:.2f}/10")
        print(f"   Visualization: {output_path}")
    
    return results


def analyze_all_afm_files(afm_dir: str = 'data/AFM', num_regions: int = 10):
    """Analyze all AFM h5 files in a directory
    
    Args:
        afm_dir: Directory containing AFM h5 files
        num_regions: Number of regions to analyze per file
    """
    # Find all h5 files
    h5_files = sorted(glob.glob(os.path.join(afm_dir, '*.h5')))
    
    if not h5_files:
        print(f"❌ No h5 files found in {afm_dir}")
        return
    
    print("\n" + "="*70)
    print(f"  SMARTSCAN - REAL AFM DATA ANALYSIS")
    print(f"  Found {len(h5_files)} AFM files")
    print("="*70)
    
    all_results = {}
    
    for h5_file in h5_files:
        results = analyze_afm_file(h5_file, num_regions=num_regions)
        all_results[os.path.basename(h5_file)] = results
    
    # Overall summary
    print("\n" + "="*70)
    print("  OVERALL SUMMARY")
    print("="*70)
    
    for filename, results in all_results.items():
        avg_quality = np.mean([r['quality'] for r in results])
        print(f"   {filename:30s} Quality: {avg_quality:.2f}/10")
    
    print("\n✅ Analysis complete!")
    print(f"   Visualizations saved to: results/afm_visualizations/")


def inspect_file_structure(h5_file_path: str):
    """Inspect the structure of an AFM h5 file"""
    print("\n" + "="*70)
    print(f"  INSPECTING: {os.path.basename(h5_file_path)}")
    print("="*70)
    inspect_h5_structure(h5_file_path, max_depth=4)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SmartScan Real AFM Data Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all AFM files in data/AFM/
  python analyze_afm.py
  
  # Analyze a specific file
  python analyze_afm.py --file data/AFM/PMN28Pt0000.h5
  
  # Inspect file structure
  python analyze_afm.py --inspect data/AFM/PMN28Pt0000.h5
  
  # Analyze with more regions
  python analyze_afm.py --regions 20
        """
    )
    
    parser.add_argument(
        '--file', 
        type=str,
        help='Analyze a specific AFM h5 file'
    )
    parser.add_argument(
        '--dir',
        type=str,
        default='data/AFM',
        help='Directory containing AFM h5 files (default: data/AFM)'
    )
    parser.add_argument(
        '--regions',
        type=int,
        default=10,
        help='Number of regions to analyze (default: 10)'
    )
    parser.add_argument(
        '--inspect',
        type=str,
        help='Inspect the structure of an h5 file'
    )
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs('results/afm_visualizations', exist_ok=True)
    
    if args.inspect:
        # Inspect file structure
        inspect_file_structure(args.inspect)
    elif args.file:
        # Analyze single file
        analyze_afm_file(args.file, num_regions=args.regions)
    else:
        # Analyze all files in directory
        analyze_all_afm_files(args.dir, num_regions=args.regions)


if __name__ == "__main__":
    main()
