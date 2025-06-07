# Analyzes the JSONL output from the benchmark script to produce a summary report.

import json
import sys
import os
from collections import defaultdict

# Optional dependencies for plotting and tables
try:
    from tabulate import tabulate
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_ENABLED = True
except ImportError:
    PLOTTING_ENABLED = False

def analyze_log_file(filepath):
    """
    Analyzes a JSONL log file to compute per-category and overall winners
    based on the Brier score, generating JSON, TXT, and plot outputs.
    """
    print(f"Analyzing log file: {filepath}")
    try:
        with open(filepath, 'r') as f:
            results = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Log file not found at '{filepath}'")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from '{filepath}'. Invalid format on line. {e}")
        sys.exit(1)

    if not results:
        print("Log file is empty. Nothing to analyze.")
        return

    # 1. Aggregate stats for Brier score, accuracy, and duration
    category_model_stats = defaultdict(lambda: defaultdict(lambda: {'brier_scores': [], 'correct_count': 0, 'total_count': 0, 'durations': []}))
    overall_model_stats = defaultdict(lambda: {'brier_scores': [], 'correct_count': 0, 'total_count': 0, 'durations': []})

    for r in results:
        model = r['model']
        true_intent = r['category']
        predicted_intent = r.get('intent', 'unknown')
        confidence = r.get('confidence', 0.0)
        duration = r.get('duration', 0.0)

        is_correct = 1 if true_intent == predicted_intent else 0
        brier_score = (confidence - is_correct)**2

        # Per-category aggregation
        cat_stats = category_model_stats[true_intent][model]
        cat_stats['brier_scores'].append(brier_score)
        cat_stats['correct_count'] += is_correct
        cat_stats['total_count'] += 1
        cat_stats['durations'].append(duration)

        # Overall aggregation
        over_stats = overall_model_stats[model]
        over_stats['brier_scores'].append(brier_score)
        over_stats['correct_count'] += is_correct
        over_stats['total_count'] += 1
        over_stats['durations'].append(duration)

    summary = {
        'per_category': {},
        'overall_winner': {},
        'model_performance_summary': []
    }
    
    # 2. Determine per-category winners (lowest Brier score wins)
    for category, models_data in category_model_stats.items():
        best_model = None
        lowest_brier_score = float('inf')

        for model, stats in models_data.items():
            avg_brier = sum(stats['brier_scores']) / len(stats['brier_scores']) if stats['brier_scores'] else float('inf')
            if avg_brier < lowest_brier_score:
                lowest_brier_score = avg_brier
                best_model = model
        
        if best_model:
            winner_stats = category_model_stats[category][best_model]
            accuracy = winner_stats['correct_count'] / winner_stats['total_count'] if winner_stats['total_count'] > 0 else 0
            avg_duration = sum(winner_stats['durations']) / len(winner_stats['durations']) if winner_stats['durations'] else 0

            summary['per_category'][category] = {
                'best_model': best_model,
                'brier_score': round(lowest_brier_score, 4),
                'accuracy': round(accuracy, 4),
                'average_duration': round(avg_duration, 2)
            }

    # 3. Determine overall winner (lowest Brier score wins)
    overall_winner_model = None
    lowest_overall_brier = float('inf')

    for model, stats in overall_model_stats.items():
        avg_brier = sum(stats['brier_scores']) / len(stats['brier_scores']) if stats['brier_scores'] else float('inf')
        if avg_brier < lowest_overall_brier:
            lowest_overall_brier = avg_brier
            overall_winner_model = model
    
    if overall_winner_model:
        winner_stats = overall_model_stats[overall_winner_model]
        accuracy = winner_stats['correct_count'] / winner_stats['total_count'] if winner_stats['total_count'] > 0 else 0
        avg_duration = sum(winner_stats['durations']) / len(winner_stats['durations']) if winner_stats['durations'] else 0

        summary['overall_winner'] = {
            'model': overall_winner_model,
            'brier_score': round(lowest_overall_brier, 4),
            'accuracy': round(accuracy, 4),
            'average_duration': round(avg_duration, 2)
        }

    # 4. Create summary table data
    table_data = []
    for model, stats in sorted(overall_model_stats.items()):
        total_count = stats['total_count']
        correct_count = stats['correct_count']
        avg_brier = sum(stats['brier_scores']) / len(stats['brier_scores']) if stats['brier_scores'] else float('inf')
        avg_duration = sum(stats['durations']) / len(stats['durations']) if stats['durations'] else 0

        table_data.append({
            'model': model,
            'correct_predictions': correct_count,
            'incorrect_predictions': total_count - correct_count,
            'accuracy': round(correct_count / total_count if total_count > 0 else 0, 4),
            'brier_score': round(avg_brier, 4),
            'average_duration': round(avg_duration, 2)
        })
    summary['model_performance_summary'] = table_data

    # 5. Create output directory and save all files
    base_name = os.path.basename(filepath).replace('.jsonl', '')
    output_dir = f"analysis_for_{base_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON summary
    json_summary_path = os.path.join(output_dir, "summary.json")
    with open(json_summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"JSON summary saved to: {json_summary_path}")

    # Save ASCII table summary
    if PLOTTING_ENABLED:
        headers = {
            'model': "Model",
            'correct_predictions': "Correct",
            'incorrect_predictions': "Incorrect",
            'accuracy': "Accuracy",
            'brier_score': "Brier Score",
            'average_duration': "Avg. Duration (s)"
        }
        winner_model_name = summary.get('overall_winner', {}).get('model', 'N/A')
        
        text_summary = tabulate(table_data, headers=headers, tablefmt="grid")
        text_summary_path = os.path.join(output_dir, "summary.txt")
        with open(text_summary_path, 'w') as f:
            f.write("LLM Intent Classification Performance Summary\n")
            f.write("="*40 + "\n\n")
            f.write(f"Overall Winner (by lowest Brier Score): {winner_model_name}\n\n")
            f.write(text_summary)
        print(f"Text summary saved to: {text_summary_path}")

        # Save Plots
        create_plots(table_data, output_dir)
    else:
        print("\nSkipping TXT and plot generation.")
        print("To enable, install optional dependencies: pip install matplotlib tabulate numpy")

def create_plots(table_data, output_dir):
    """Generates and saves bar plots for key performance metrics."""
    if not PLOTTING_ENABLED:
        print("Plotting libraries not found. Skipping plot generation.")
        return

    models = [d['model'] for d in table_data]
    if not models:
        print("No data to plot.")
        return

    # Accuracy Plot (Higher is better)
    accuracy = [d['accuracy'] for d in table_data]
    plt.figure(figsize=(12, 7))
    bars = plt.bar(models, accuracy, color='lightgreen')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison: Accuracy (Higher is Better)')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.bar_label(bars, fmt='%.2f')
    plt.tight_layout()
    accuracy_plot_path = os.path.join(output_dir, 'accuracy_comparison.png')
    plt.savefig(accuracy_plot_path)
    plt.close()
    print(f"Accuracy plot saved to: {accuracy_plot_path}")

    # Combined Brier Score and Duration Plot
    brier_scores = [d['brier_score'] for d in table_data]
    durations = [d['average_duration'] for d in table_data]
    
    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Bar for Brier Score
    color1 = 'skyblue'
    ax1.set_ylabel('Brier Score', color=color1)
    bar1 = ax1.bar(x - width/2, brier_scores, width, label='Brier Score', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    color2 = 'salmon'
    ax2.set_ylabel('Average Duration (s)', color=color2)
    bar2 = ax2.bar(x + width/2, durations, width, label='Avg. Duration (s)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title('Model Comparison: Brier Score & Duration (Lower is Better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right")

    ax1.bar_label(bar1, fmt='%.4f', padding=3)
    ax2.bar_label(bar2, fmt='%.2f s', padding=3)

    fig.tight_layout()
    
    brier_duration_plot_path = os.path.join(output_dir, 'brier_duration_comparison.png')
    plt.savefig(brier_duration_plot_path)
    plt.close()
    print(f"Combined Brier/duration plot saved to: {brier_duration_plot_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_results.py <path_to_jsonl_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    analyze_log_file(log_file) 