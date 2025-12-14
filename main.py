'''SmartScan Demo Script'''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from src.adaptive_scanner import AdaptiveScanner, compare_methods


def create_comparison_plots(results_trad, results_adapt):
    '''Generate publication-quality comparison plots'''

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

    # Title
    fig.suptitle('SmartScan: Real-Time Adaptive AFM Scanning',
                 fontsize=20, fontweight='bold')

    # Plot 1: Time comparison
    ax1 = fig.add_subplot(gs[0, 0])
    methods = ['Traditional', 'SmartScan']
    times = [results_trad['total_time'], results_adapt['total_time']]
    colors = ['#FF6B6B', '#4ECDC4']
    bars = ax1.bar(methods, times, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Total Scan Time (s)', fontsize=12)
    ax1.set_title('A) Scan Speed', fontsize=14, fontweight='bold')

    # Add percentage on bars
    improvement = (times[0] - times[1]) / times[0] * 100
    ax1.text(1, times[1], f'-{improvement:.0f}%',
             ha='center', va='bottom', fontsize=14, fontweight='bold', color='green')

    # Plot 2: Quality comparison
    ax2 = fig.add_subplot(gs[0, 1])
    qualities = [results_trad['avg_quality'], results_adapt['avg_quality']]
    bars = ax2.bar(methods, qualities, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Average Quality Score', fontsize=12)
    ax2.set_ylim(0, 10)
    ax2.axhline(7, color='gray', linestyle='--', alpha=0.5, label='Target Quality')
    ax2.set_title('B) Image Quality', fontsize=14, fontweight='bold')
    ax2.legend()

    # Plot 3: Time per region
    ax3 = fig.add_subplot(gs[0, 2])
    x = range(len(results_trad['time_log']))
    ax3.plot(x, results_trad['time_log'], 'o-', label='Traditional',
             color='#FF6B6B', linewidth=2, markersize=8)
    ax3.plot(x, results_adapt['time_log'], 's-', label='SmartScan',
             color='#4ECDC4', linewidth=2, markersize=8)
    ax3.set_xlabel('Region Number', fontsize=12)
    ax3.set_ylabel('Scan Time (s)', fontsize=12)
    ax3.set_title('C) Time per Region', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Plot 4: Quality progression
    ax4 = fig.add_subplot(gs[1, :])
    trad_qualities = [a['quality'] for a in results_trad['analyses']]
    adapt_qualities = [a['quality'] for a in results_adapt['analyses']]

    x = range(len(trad_qualities))
    ax4.plot(x, trad_qualities, 'o-', label='Traditional',
             color='#FF6B6B', linewidth=3, markersize=10, alpha=0.7)
    ax4.plot(x, adapt_qualities, 's-', label='SmartScan',
             color='#4ECDC4', linewidth=3, markersize=10, alpha=0.7)
    ax4.axhline(7, color='gray', linestyle='--', alpha=0.5, label='Target Quality')
    ax4.set_xlabel('Region Number', fontsize=14)
    ax4.set_ylabel('Quality Score', fontsize=14)
    ax4.set_title('D) Quality Throughout Scan', fontsize=16, fontweight='bold')
    ax4.legend(fontsize=12)
    ax4.grid(alpha=0.3)
    ax4.set_ylim(0, 10)

    # Plot 5: Parameter adaptation
    ax5 = fig.add_subplot(gs[2, :])
    speeds = [p['speed'] for p in results_adapt['param_log']]
    ax5.plot(x, speeds, 'o-', color='#4ECDC4', linewidth=3, markersize=10)
    ax5.fill_between(x, speeds, alpha=0.3, color='#4ECDC4')
    ax5.set_xlabel('Region Number', fontsize=14)
    ax5.set_ylabel('Scan Speed (µm/s)', fontsize=14)
    ax5.set_title('E) Real-Time Parameter Adaptation', fontsize=16, fontweight='bold')
    ax5.grid(alpha=0.3)

    # Add annotations
    for i, (q, s) in enumerate(zip(adapt_qualities, speeds)):
        if q < 6:
            ax5.annotate('Low quality\n→ Slowing down',
                        xy=(i, s), xytext=(i+1.2, s+5),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                        fontsize=10, ha='center', color='red', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', alpha=0.8))
            break

    plt.savefig('results/smartscan_benchmark.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved visualization: results/smartscan_benchmark.png")
    plt.show()


def run_demo():
    '''Run complete demo'''
    print("\n" + "="*70)
    print("  SMARTSCAN: Real-Time Adaptive AFM Scanning")
    print("  Microscopy Hackathon 2025")
    print("="*70 + "\n")

    # Run comparison
    results_trad, results_adapt = compare_methods(num_regions=10)

    # Create plots
    create_comparison_plots(results_trad, results_adapt)

    # Print summary
    print("\n" + "="*70)
    print("📋 SUMMARY FOR PRESENTATION")
    print("="*70)
    print(f"\n✅ SmartScan Performance:")
    time_saved = results_trad['total_time'] - results_adapt['total_time']
    print(f"   • {time_saved:.0f}s faster ({time_saved/results_trad['total_time']*100:.0f}% improvement)")
    quality_gain = results_adapt['avg_quality'] - results_trad['avg_quality']
    if quality_gain > 0:
        print(f"   • {quality_gain:.1f} points higher quality")
    print(f"   • Automatically adjusted parameters {sum(1 for a in results_adapt['analyses'] if a['should_adjust'])} times")
    print(f"\n💡 Impact: Saves {time_saved:.0f}s per sample")
    print(f"   → {time_saved/60:.0f} minutes per day")
    print(f"   → {time_saved/60*5:.0f} minutes per week")
    print(f"   → Worth $${(time_saved/60*5) * 8.33:.0f}/week at $500/hr microscope time!")

    print("\n✅ Ready for presentation!")


if __name__ == "__main__":
    run_demo()