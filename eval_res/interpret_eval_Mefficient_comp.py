import json
import pandas as pd

# Load comprehensive and baseline metrics
with open("validation_metrics_comprehensive.json", "r") as f:
    mefficient_data = json.load(f)

with open("validation_metrics.json", "r") as f:
    baseline_data = json.load(f)
    baseline_data = baseline_data["per_class_average_precision"]

label_map = pd.read_excel("../imat_data/relabel-split.xlsx", engine='openpyxl')

def print_overall_comparison():
    print("=== COMPREHENSIVE MODEL vs BASELINE COMPARISON ===\n")
    for attribute, metrics in mefficient_data.items():
        print(f"\n{attribute.upper()}:")
        if "average_precision_macro" in metrics:
            print(f"  Macro AP - Comprehensive: {metrics['average_precision_macro']:.4f}")
            print(f"  Macro AP - Baseline: {sum(baseline_data) / len(baseline_data):.4f}")
        if "f1_macro" in metrics:
            print(f"  F1 Macro - Comprehensive: {metrics['f1_macro']:.4f}")
        if "accuracy" in metrics:
            print(f"  Accuracy - Comprehensive: {metrics['accuracy']:.4f}")
        if "top3_accuracy" in metrics:
            print(f"  Top-3 Accuracy - Comprehensive: {metrics['top3_accuracy']:.4f}")
        if "subset_accuracy" in metrics:
            print(f"  Subset Accuracy - Comprehensive: {metrics['subset_accuracy']:.4f}")
        if "precision_at_3" in metrics:
            print(f"  Precision@3 - Comprehensive: {metrics['precision_at_3']:.4f}")
        print()

def per_class_comparison():
    print("=== PER-CLASS COMPARISON ===\n")
    grouped_mefficient = label_map.groupby("taskName")
    for category, group in grouped_mefficient:
        if category in mefficient_data and "average_precision_per_class" in mefficient_data[category]:
            comprehensive_aps = mefficient_data[category]["average_precision_per_class"]
            print(f"\nCategory: {category}")
            print(f"Number of classes: {len(comprehensive_aps)}")
            print("-" * 80)
            for index, row in group.iterrows():
                label_name = row['labelName']
                comprehensive_ap = comprehensive_aps[row['group_label']]
                baseline_ap = baseline_data[row['labelId_new']]
                improvement = comprehensive_ap - baseline_ap
                improvement_pct = (improvement / baseline_ap * 100) if baseline_ap > 0 else 0
                print(f"Label: {label_name}")
                print(f"  Comprehensive AP: {comprehensive_ap:.4f}")
                print(f"  Baseline AP: {baseline_ap:.4f}")
                print(f"  Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")
                print()

def summary_statistics():
    print("=== SUMMARY STATISTICS ===\n")
    grouped_mefficient = label_map.groupby("taskName")
    improvements = []
    comprehensive_scores = []
    baseline_scores = []
    for category, group in grouped_mefficient:
        if category in mefficient_data and "average_precision_per_class" in mefficient_data[category]:
            comprehensive_aps = mefficient_data[category]["average_precision_per_class"]
            for index, row in group.iterrows():
                comprehensive_ap = comprehensive_aps[row['group_label']]
                baseline_ap = baseline_data[row['labelId_new']]
                comprehensive_scores.append(comprehensive_ap)
                baseline_scores.append(baseline_ap)
                improvements.append(comprehensive_ap - baseline_ap)
    print(f"Total classes compared: {len(improvements)}")
    print(f"Average Comprehensive AP: {sum(comprehensive_scores) / len(comprehensive_scores):.4f}")
    print(f"Average Baseline AP: {sum(baseline_scores) / len(baseline_scores):.4f}")
    print(f"Average Improvement: {sum(improvements) / len(improvements):.4f}")
    print(f"Classes with improvement: {sum(1 for imp in improvements if imp > 0)} / {len(improvements)}")
    print(f"Classes with degradation: {sum(1 for imp in improvements if imp < 0)} / {len(improvements)}")
    print(f"Classes unchanged: {sum(1 for imp in improvements if imp == 0)} / {len(improvements)}")

def best_worst_classes():
    print("=== BEST AND WORST PERFORMING CLASSES ===\n")
    grouped_mefficient = label_map.groupby("taskName")
    class_performances = []
    for category, group in grouped_mefficient:
        if category in mefficient_data and "average_precision_per_class" in mefficient_data[category]:
            comprehensive_aps = mefficient_data[category]["average_precision_per_class"]
            for index, row in group.iterrows():
                label_name = row['labelName']
                comprehensive_ap = comprehensive_aps[row['group_label']]
                baseline_ap = baseline_data[row['labelId_new']]
                improvement = comprehensive_ap - baseline_ap
                class_performances.append({
                    'category': category,
                    'label': label_name,
                    'comprehensive_ap': comprehensive_ap,
                    'baseline_ap': baseline_ap,
                    'improvement': improvement
                })
    class_performances.sort(key=lambda x: x['improvement'], reverse=True)
    print("Top 10 Most Improved Classes:")
    for i, perf in enumerate(class_performances[:10]):
        print(f"{i+1}. {perf['label']} ({perf['category']})")
        print(f"   Comprehensive: {perf['comprehensive_ap']:.4f}, Baseline: {perf['baseline_ap']:.4f}")
        print(f"   Improvement: {perf['improvement']:+.4f}")
        print()
    print("\nBottom 10 Most Degraded Classes:")
    for i, perf in enumerate(class_performances[-10:]):
        print(f"{i+1}. {perf['label']} ({perf['category']})")
        print(f"   Comprehensive: {perf['comprehensive_ap']:.4f}, Baseline: {perf['baseline_ap']:.4f}")
        print(f"   Degradation: {perf['improvement']:+.4f}")
        print()

def main():
    print_overall_comparison()
    per_class_comparison()
    summary_statistics()
    best_worst_classes()

if __name__ == "__main__":
    main() 