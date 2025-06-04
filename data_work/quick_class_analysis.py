#!/usr/bin/env python3
"""
Quick Class Breakdown Analysis Script

This script provides a fast way to analyze class distribution in your train/val datasets.
It can work with both original and relabeled annotation files.

Usage:
    python quick_class_analysis.py
"""

import json
import pandas as pd
import numpy as np
from collections import Counter
import argparse
import os

def load_annotations(file_path):
    """Load annotation file and return annotations list"""
    print(f"Loading {file_path}...")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data['annotations']
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def analyze_annotations(annotations, dataset_name):
    """Analyze annotations and return summary statistics"""
    print(f"Analyzing {dataset_name} dataset...")
    
    if not annotations:
        return None
    
    # Basic statistics
    total_images = len(annotations)
    label_counts = Counter()
    labels_per_image = []
    
    # Count labels
    for annotation in annotations:
        if annotation.get('labelId') is None:
            continue
        
        labels = annotation['labelId']
        if isinstance(labels, list):
            labels_per_image.append(len(labels))
            for label in labels:
                label_counts[label] += 1
        else:
            # Single label case
            labels_per_image.append(1)
            label_counts[labels] += 1
    
    # Calculate statistics
    stats = {
        'dataset': dataset_name,
        'total_images': total_images,
        'images_with_labels': len(labels_per_image),
        'total_labels': sum(labels_per_image),
        'unique_labels': len(label_counts),
        'avg_labels_per_image': np.mean(labels_per_image) if labels_per_image else 0,
        'median_labels_per_image': np.median(labels_per_image) if labels_per_image else 0,
        'min_labels_per_image': np.min(labels_per_image) if labels_per_image else 0,
        'max_labels_per_image': np.max(labels_per_image) if labels_per_image else 0,
        'label_counts': dict(label_counts)
    }
    
    return stats

def load_label_mapping(label_file_path):
    """Load label mapping from Excel file"""
    try:
        df = pd.read_excel(label_file_path, engine='openpyxl')
        print(f"Loaded label mapping with {len(df)} labels")
        return df
    except Exception as e:
        print(f"Warning: Could not load label mapping from {label_file_path}: {e}")
        return None

def get_label_name(label_id, label_map_df):
    """Get label name from ID"""
    if label_map_df is None:
        return f"Label_{label_id}"
    
    # Try both labelId and labelId_new columns
    for col in ['labelId_new', 'labelId']:
        if col in label_map_df.columns:
            match = label_map_df[label_map_df[col] == label_id]
            if len(match) > 0:
                return match['labelName'].iloc[0]
    
    return f"Unknown_{label_id}"

def print_summary(train_stats, val_stats):
    """Print summary comparison table"""
    print("\n" + "="*60)
    print("DATASET SUMMARY COMPARISON")
    print("="*60)
    
    if train_stats is None or val_stats is None:
        print("Error: Unable to generate summary - missing data")
        return
    
    # Create comparison table
    comparison = pd.DataFrame({
        'Metric': [
            'Total Images',
            'Images with Labels',
            'Total Label Annotations',
            'Unique Labels',
            'Avg Labels per Image',
            'Median Labels per Image',
            'Min Labels per Image',
            'Max Labels per Image'
        ],
        'Training': [
            train_stats['total_images'],
            train_stats['images_with_labels'],
            train_stats['total_labels'],
            train_stats['unique_labels'],
            f"{train_stats['avg_labels_per_image']:.2f}",
            f"{train_stats['median_labels_per_image']:.1f}",
            train_stats['min_labels_per_image'],
            train_stats['max_labels_per_image']
        ],
        'Validation': [
            val_stats['total_images'],
            val_stats['images_with_labels'],
            val_stats['total_labels'],
            val_stats['unique_labels'],
            f"{val_stats['avg_labels_per_image']:.2f}",
            f"{val_stats['median_labels_per_image']:.1f}",
            val_stats['min_labels_per_image'],
            val_stats['max_labels_per_image']
        ]
    })
    
    print(comparison.to_string(index=False))

def create_class_breakdown(train_stats, val_stats, label_map_df, top_n=30):
    """Create detailed class breakdown"""
    print(f"\n" + "="*60)
    print(f"TOP {top_n} CLASSES BREAKDOWN")
    print("="*60)
    
    # Combine all labels
    all_labels = set()
    if train_stats:
        all_labels.update(train_stats['label_counts'].keys())
    if val_stats:
        all_labels.update(val_stats['label_counts'].keys())
    
    # Create breakdown
    breakdown_data = []
    for label_id in all_labels:
        train_count = train_stats['label_counts'].get(label_id, 0) if train_stats else 0
        val_count = val_stats['label_counts'].get(label_id, 0) if val_stats else 0
        total_count = train_count + val_count
        
        breakdown_data.append({
            'Label_ID': label_id,
            'Label_Name': get_label_name(label_id, label_map_df),
            'Train_Count': train_count,
            'Val_Count': val_count,
            'Total_Count': total_count,
            'Train_Pct': f"{100*train_count/total_count:.1f}%" if total_count > 0 else "0%"
        })
    
    # Sort by total count and show top N
    breakdown_df = pd.DataFrame(breakdown_data)
    breakdown_df = breakdown_df.sort_values('Total_Count', ascending=False)
    
    # Show top classes
    top_classes = breakdown_df.head(top_n)
    print(top_classes[['Label_Name', 'Train_Count', 'Val_Count', 'Total_Count', 'Train_Pct']].to_string(index=False))
    
    # Save full breakdown
    breakdown_df.to_csv('quick_class_breakdown.csv', index=False)
    print(f"\nFull breakdown saved to 'quick_class_breakdown.csv'")
    
    return breakdown_df

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Quick class breakdown analysis')
    parser.add_argument('--train', default='imat_data/train_annos_relabel.json', 
                       help='Path to training annotations')
    parser.add_argument('--val', default='imat_data/val_annos_relabel.json',
                       help='Path to validation annotations')
    parser.add_argument('--labels', default='imat_data/relabel.xlsx',
                       help='Path to label mapping file')
    parser.add_argument('--top', type=int, default=30,
                       help='Number of top classes to show')
    
    args = parser.parse_args()
    
    print("QUICK CLASS BREAKDOWN ANALYSIS")
    print("="*60)
    
    # Load data
    train_annotations = load_annotations(args.train)
    val_annotations = load_annotations(args.val)
    label_map_df = load_label_mapping(args.labels)
    
    # Analyze datasets
    train_stats = analyze_annotations(train_annotations, "Training") if train_annotations else None
    val_stats = analyze_annotations(val_annotations, "Validation") if val_annotations else None
    
    # Print results
    if train_stats and val_stats:
        print_summary(train_stats, val_stats)
        create_class_breakdown(train_stats, val_stats, label_map_df, args.top)
    elif train_stats:
        print("Only training data available")
        print(f"Training: {train_stats}")
    elif val_stats:
        print("Only validation data available") 
        print(f"Validation: {val_stats}")
    else:
        print("No valid data found!")
        return
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 