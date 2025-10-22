"""
Enhanced Coverage Area Visualization and Analysis
==================================================
Detailed visualization of how offensive players create coverage stress
and maximize field control against zone defenses.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
import seaborn as sns
from .coverage_area_analyzer import CoverageAreaAnalyzer
from .zone_coverage import ZoneCoverage
from typing import Dict, List, Tuple

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class CoverageVisualizer:
    """Visualize coverage area maximization concepts"""
    
    def __init__(self, analyzer: CoverageAreaAnalyzer):
        self.analyzer = analyzer
        self.field_width = 53.3
        self.los = analyzer.los
        
    def plot_zone_coverage_map(self, frame_id: int, coverage_type: str = 'cover3'):
        """Plot the field with zone responsibilities overlaid"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Get frame data
        off_frame = self.analyzer.offense_data[self.analyzer.offense_data['frame_id'] == frame_id]
        def_frame = self.analyzer.defense_data[self.analyzer.defense_data['frame_id'] == frame_id]
        
        # Get zones
        if coverage_type == 'cover3':
            zones = ZoneCoverage.get_cover3_zones(self.field_width, self.los)
        else:
            zones = ZoneCoverage.get_cover2_zones(self.field_width, self.los)
        
        # Left plot: Zone definitions and player positions
        ax1.set_title(f'{coverage_type.upper()} Zone Coverage - Frame {frame_id}', fontsize=14, fontweight='bold')
        
        # Draw field
        ax1.axhline(y=0, color='white', linewidth=2)
        ax1.axhline(y=self.field_width, color='white', linewidth=2)
        ax1.axvline(x=self.los, color='yellow', linewidth=2, label='Line of Scrimmage')
        
        # Draw zones with transparency
        colors = plt.cm.Set3(np.linspace(0, 1, len(zones)))
        for (zone_name, zone_def), color in zip(zones.items(), colors):
            rect = Rectangle(
                (zone_def['x_range'][0], zone_def['y_range'][0]),
                zone_def['x_range'][1] - zone_def['x_range'][0],
                zone_def['y_range'][1] - zone_def['y_range'][0],
                alpha=0.3, facecolor=color, edgecolor='black', linewidth=1
            )
            ax1.add_patch(rect)
            
            # Add zone label
            cx = (zone_def['x_range'][0] + zone_def['x_range'][1]) / 2
            cy = (zone_def['y_range'][0] + zone_def['y_range'][1]) / 2
            ax1.text(cx, cy, zone_name.replace('_', '\n'), 
                    ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Plot offensive players
        for _, player in off_frame.iterrows():
            if player['player_position'] in ['WR', 'TE', 'RB']:
                ax1.scatter(player['x'], player['y'], s=150, c='blue', 
                          marker='^', edgecolor='white', linewidth=2, zorder=5)
                ax1.text(player['x'], player['y']-1.5, player['player_name'].split()[-1],
                        ha='center', fontsize=8, fontweight='bold')
            elif player['player_position'] == 'QB':
                ax1.scatter(player['x'], player['y'], s=150, c='darkblue', 
                          marker='s', edgecolor='white', linewidth=2, zorder=5)
        
        # Plot defensive players
        for _, player in def_frame.iterrows():
            ax1.scatter(player['x'], player['y'], s=150, c='red', 
                      marker='o', edgecolor='white', linewidth=2, zorder=5)
            ax1.text(player['x'], player['y']+1.5, 
                    f"{player['player_name'].split()[-1]}\n({player['player_position']})",
                    ha='center', fontsize=7)
        
        ax1.set_xlim(self.los - 5, self.los + 35)
        ax1.set_ylim(-2, self.field_width + 2)
        ax1.set_xlabel('Field Position (yards)', fontsize=11)
        ax1.set_ylabel('Field Width (yards)', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Zone stress heatmap
        ax2.set_title('Zone Stress Analysis', fontsize=14, fontweight='bold')
        
        zone_stress = self.analyzer.calculate_zone_stress(frame_id, coverage_type)
        
        # Create stress grid
        x_bins = np.linspace(self.los, self.los + 30, 7)
        y_bins = np.linspace(0, self.field_width, 7)
        stress_grid = np.zeros((len(y_bins)-1, len(x_bins)-1))
        
        for zone_name, stress_data in zone_stress.items():
            zone_def = zones[zone_name]
            x_idx = np.digitize((zone_def['x_range'][0] + zone_def['x_range'][1])/2, x_bins) - 1
            y_idx = np.digitize((zone_def['y_range'][0] + zone_def['y_range'][1])/2, y_bins) - 1
            
            if 0 <= x_idx < stress_grid.shape[1] and 0 <= y_idx < stress_grid.shape[0]:
                stress_grid[y_idx, x_idx] = stress_data['stress_ratio']
        
        # Plot heatmap
        im = ax2.imshow(stress_grid, cmap='YlOrRd', aspect='auto', 
                       extent=[self.los, self.los+30, 0, self.field_width],
                       vmin=0, vmax=2, origin='lower')
        
        # Add player positions on heatmap
        for _, player in off_frame.iterrows():
            if player['player_position'] in ['WR', 'TE', 'RB']:
                ax2.scatter(player['x'], player['y'], s=100, c='blue', 
                          marker='^', edgecolor='white', linewidth=2, zorder=5)
        
        for _, player in def_frame.iterrows():
            ax2.scatter(player['x'], player['y'], s=100, c='white', 
                      marker='o', edgecolor='black', linewidth=2, zorder=5)
        
        ax2.set_xlim(self.los - 5, self.los + 35)
        ax2.set_ylim(-2, self.field_width + 2)
        ax2.set_xlabel('Field Position (yards)', fontsize=11)
        ax2.set_ylabel('Field Width (yards)', fontsize=11)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Zone Stress Ratio', fontsize=10)
        
        plt.suptitle('Coverage Area Maximization Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig
    
    def plot_route_synergy_analysis(self, frame_id: int):
        """Visualize route synergy and spacing concepts"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Get receiver data
        off_frame = self.analyzer.offense_data[
            (self.analyzer.offense_data['frame_id'] == frame_id) & 
            (self.analyzer.offense_data['player_position'].isin(['WR', 'TE', 'RB']))
        ]
        
        receiver_ids = off_frame['nfl_id'].unique()
        synergy = self.analyzer.calculate_route_synergy(frame_id, receiver_ids.tolist())
        
        # Plot 1: Route spacing and coverage area
        ax = axes[0, 0]
        ax.set_title('Route Spacing & Coverage Area', fontsize=12, fontweight='bold')
        
        positions = off_frame[['x', 'y']].values
        
        # Draw convex hull if possible
        if len(positions) >= 3:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(positions)
            for simplex in hull.simplices:
                ax.plot(positions[simplex, 0], positions[simplex, 1], 'b-', alpha=0.3, linewidth=2)
            
            # Fill the hull
            hull_points = positions[hull.vertices]
            from matplotlib.patches import Polygon
            poly = Polygon(hull_points, alpha=0.15, facecolor='blue', edgecolor='blue')
            ax.add_patch(poly)
        
        # Plot receivers
        for _, player in off_frame.iterrows():
            ax.scatter(player['x'], player['y'], s=200, c='blue', 
                      marker='^', edgecolor='white', linewidth=2, zorder=5)
            ax.text(player['x'], player['y']-1.5, player['player_name'].split()[-1],
                    ha='center', fontsize=9, fontweight='bold')
        
        # Add spacing lines between all receivers
        from itertools import combinations
        for (i, p1), (j, p2) in combinations(enumerate(positions), 2):
            dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g--', alpha=0.5, linewidth=1)
            mid_x, mid_y = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
            ax.text(mid_x, mid_y, f'{dist:.1f}y', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax.set_xlabel('Field Position (yards)')
        ax.set_ylabel('Field Width (yards)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min(positions[:, 0])-5, max(positions[:, 0])+5)
        ax.set_ylim(min(positions[:, 1])-5, max(positions[:, 1])+5)
        
        # Add metrics text
        metrics_text = f"Coverage Area: {synergy['components']['coverage_area']:.1f} sq yards\n"
        metrics_text += f"H-Stretch: {synergy['components']['horizontal_stretch']:.1f}y | "
        metrics_text += f"V-Stretch: {synergy['components']['vertical_stretch']:.1f}y"
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               verticalalignment='top', fontsize=9)
        
        # Plot 2: Defender leverage analysis
        ax = axes[0, 1]
        ax.set_title('Defender Leverage & Burden', fontsize=12, fontweight='bold')
        
        def_frame = self.analyzer.defense_data[self.analyzer.defense_data['frame_id'] == frame_id]
        
        # Plot field
        ax.axvline(x=self.los, color='yellow', linewidth=2, alpha=0.5)
        
        # Plot all players
        for _, player in off_frame.iterrows():
            ax.scatter(player['x'], player['y'], s=150, c='blue', 
                      marker='^', edgecolor='white', linewidth=2, zorder=5)
        
        # Plot defenders with burden coloring
        burden_scores = []
        for _, defender in def_frame.iterrows():
            burden = self.analyzer.calculate_coverage_burden(frame_id, defender['nfl_id'])
            burden_scores.append(burden['burden_score'])
            
            color_intensity = plt.cm.Reds(0.3 + 0.7 * burden['burden_score'])
            ax.scatter(defender['x'], defender['y'], s=200, c=[color_intensity],
                      marker='o', edgecolor='black', linewidth=2, zorder=4)
            
            # Draw coverage radius
            circle = Circle((defender['x'], defender['y']), 7, 
                          fill=False, edgecolor=color_intensity, linewidth=1, alpha=0.5)
            ax.add_patch(circle)
            
            # Label with burden score
            ax.text(defender['x'], defender['y']+2, 
                   f"{defender['player_position']}\n{burden['burden_score']:.2f}",
                   ha='center', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Field Position (yards)')
        ax.set_ylabel('Field Width (yards)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(self.los-5, self.los+30)
        ax.set_ylim(-2, self.field_width+2)
        
        # Plot 3: Zone stress timeline
        ax = axes[1, 0]
        ax.set_title('Zone Stress Evolution', fontsize=12, fontweight='bold')
        
        frames_to_analyze = range(1, min(25, self.analyzer.data['frame_id'].max()+1), 2)
        zone_stress_timeline = {zone: [] for zone in ['deep_left', 'deep_middle', 'deep_right', 
                                                      'hook_curl_left', 'middle_hole', 'hook_curl_right']}
        
        for f in frames_to_analyze:
            stress = self.analyzer.calculate_zone_stress(f, 'cover3')
            for zone in zone_stress_timeline:
                if zone in stress:
                    zone_stress_timeline[zone].append(stress[zone]['stress_ratio'])
                else:
                    zone_stress_timeline[zone].append(0)
        
        for zone, values in zone_stress_timeline.items():
            if max(values) > 0.1:  # Only plot zones with some stress
                ax.plot(frames_to_analyze, values, marker='o', label=zone.replace('_', ' ').title(), linewidth=2)
        
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Stress Threshold')
        ax.set_xlabel('Frame ID')
        ax.set_ylabel('Zone Stress Ratio')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Synergy components breakdown
        ax = axes[1, 1]
        ax.set_title('Route Synergy Components', fontsize=12, fontweight='bold')
        
        components = synergy['components']
        labels = ['Spacing\nQuality', 'Coverage\nArea', 'Horizontal\nStretch', 
                 'Vertical\nStretch', 'Concept\nBonus']
        values = [
            components['spacing_quality'],
            min(1, components['coverage_area'] / 400),
            min(1, components['horizontal_stretch'] / 40),
            min(1, components['vertical_stretch'] / 20),
            components['concept_bonus']
        ]
        
        bars = ax.bar(labels, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax.set_ylabel('Score (0-1)', fontsize=10)
        ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add overall synergy score
        overall_score = synergy['synergy_score']
        ax.text(0.5, 0.95, f'Overall Synergy Score: {overall_score:.3f}',
               transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        if components['concept_identified'] != 'unknown':
            ax.text(0.5, 0.85, f'Concept: {components["concept_identified"].upper()}',
                   transform=ax.transAxes, ha='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Route Synergy Analysis - Frame {frame_id}', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig
    
    def create_summary_report(self, frame_id: int):
        """Create a comprehensive summary visualization"""
        
        results = self.analyzer.analyze_full_play(frame_id, 'cover3')
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main field view
        ax_main = fig.add_subplot(gs[:2, :2])
        self._plot_field_overview(ax_main, frame_id, results)
        
        # Metrics dashboard
        ax_metrics = fig.add_subplot(gs[:2, 2])
        self._plot_metrics_dashboard(ax_metrics, results)
        
        # Zone stress bars
        ax_zones = fig.add_subplot(gs[2, :2])
        self._plot_zone_stress_bars(ax_zones, results)
        
        # Defender burden pie
        ax_burden = fig.add_subplot(gs[2, 2])
        self._plot_defender_burden_pie(ax_burden, results)
        
        plt.suptitle(f'Coverage Area Maximization Summary - Frame {frame_id}',
                    fontsize=18, fontweight='bold', y=1.02)
        
        return fig
    
    def _plot_field_overview(self, ax, frame_id, results):
        """Plot main field view with all elements"""
        ax.set_title('Field Overview', fontsize=12, fontweight='bold')
        
        off_frame = self.analyzer.offense_data[self.analyzer.offense_data['frame_id'] == frame_id]
        def_frame = self.analyzer.defense_data[self.analyzer.defense_data['frame_id'] == frame_id]
        
        # Draw field elements
        ax.axhline(y=0, color='white', linewidth=2)
        ax.axhline(y=self.field_width, color='white', linewidth=2)
        ax.axvline(x=self.los, color='yellow', linewidth=2, alpha=0.7, label='LOS')
        
        # Draw stressed zones
        zones = ZoneCoverage.get_cover3_zones(self.field_width, self.los)
        for zone_name, zone_def in zones.items():
            stress = results['zone_stress'][zone_name]['stress_ratio']
            if stress > 0.5:
                color_intensity = plt.cm.YlOrRd(min(1, stress/2))
                rect = Rectangle(
                    (zone_def['x_range'][0], zone_def['y_range'][0]),
                    zone_def['x_range'][1] - zone_def['x_range'][0],
                    zone_def['y_range'][1] - zone_def['y_range'][0],
                    alpha=0.4, facecolor=color_intensity, edgecolor='darkred', linewidth=1
                )
                ax.add_patch(rect)
        
        # Plot players
        for _, player in off_frame.iterrows():
            if player['player_position'] in ['WR', 'TE', 'RB']:
                ax.scatter(player['x'], player['y'], s=180, c='blue', 
                          marker='^', edgecolor='white', linewidth=2, zorder=5)
                ax.text(player['x'], player['y']-1.8, player['player_name'].split()[-1],
                        ha='center', fontsize=8, fontweight='bold', color='blue')
            elif player['player_position'] == 'QB':
                ax.scatter(player['x'], player['y'], s=180, c='darkblue', 
                          marker='s', edgecolor='white', linewidth=2, zorder=5)
        
        for _, player in def_frame.iterrows():
            burden = results['defender_burdens'][player['nfl_id']]
            color_intensity = plt.cm.Reds(0.3 + 0.7 * burden['burden_score'])
            ax.scatter(player['x'], player['y'], s=180, c=[color_intensity],
                      marker='o', edgecolor='black', linewidth=2, zorder=4)
            ax.text(player['x'], player['y']+1.8, player['player_position'],
                    ha='center', fontsize=8, color='darkred')
        
        # Add ball landing spot
        ax.scatter(self.analyzer.data['ball_land_x'].iloc[0], 
                  self.analyzer.data['ball_land_y'].iloc[0],
                  s=100, c='yellow', marker='*', edgecolor='black', 
                  linewidth=2, zorder=6, label='Ball Landing')
        
        ax.set_xlim(self.los-5, self.los+35)
        ax.set_ylim(-2, self.field_width+2)
        ax.set_xlabel('Field Position (yards)')
        ax.set_ylabel('Field Width (yards)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    def _plot_metrics_dashboard(self, ax, results):
        """Plot key metrics dashboard"""
        ax.set_title('Performance Metrics', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        metrics = [
            ('Offensive\nAdvantage', results['offensive_advantage'], 'green'),
            ('Route\nSynergy', results['summary']['synergy_score'], 'blue'),
            ('Avg Defender\nBurden', results['summary']['average_defender_burden'], 'red'),
            ('Zone Stress\n(Normalized)', min(1, results['summary']['total_zone_stress']/10), 'orange')
        ]
        
        y_pos = 0.9
        for label, value, color in metrics:
            # Draw progress bar
            bar_width = 0.6
            bar_start = 0.2
            
            # Background
            rect_bg = Rectangle((bar_start, y_pos-0.03), bar_width, 0.06,
                               facecolor='lightgray', edgecolor='black')
            ax.add_patch(rect_bg)
            
            # Progress
            rect = Rectangle((bar_start, y_pos-0.03), bar_width*value, 0.06,
                           facecolor=color, alpha=0.7, edgecolor='black')
            ax.add_patch(rect)
            
            # Label and value
            ax.text(0.15, y_pos, label, ha='right', va='center', fontsize=9, fontweight='bold')
            ax.text(0.85, y_pos, f'{value:.2f}', ha='left', va='center', fontsize=10, fontweight='bold')
            
            y_pos -= 0.15
        
        # Add summary stats
        y_pos -= 0.1
        ax.text(0.5, y_pos, 'Summary Statistics', ha='center', fontsize=10, 
               fontweight='bold')
        ax.plot([0.3, 0.7], [y_pos-0.02, y_pos-0.02], 'k-', linewidth=1)  # Underline
        
        y_pos -= 0.08
        stats = [
            f'Stressed Zones: {results["summary"]["stressed_zones"]}',
            f'Overloaded Defenders: {results["summary"]["overloaded_defenders"]}',
            f'Total Zone Stress: {results["summary"]["total_zone_stress"]:.1f}'
        ]
        
        for stat in stats:
            ax.text(0.5, y_pos, stat, ha='center', fontsize=9)
            y_pos -= 0.06
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    def _plot_zone_stress_bars(self, ax, results):
        """Plot zone stress as horizontal bars"""
        ax.set_title('Zone Stress Distribution', fontsize=12, fontweight='bold')
        
        zones = []
        stress_values = []
        colors = []
        
        for zone_name, stress_data in results['zone_stress'].items():
            zones.append(zone_name.replace('_', ' ').title())
            stress_values.append(stress_data['stress_ratio'])
            
            if stress_data['stress_ratio'] > 1.5:
                colors.append('darkred')
            elif stress_data['stress_ratio'] > 1.0:
                colors.append('orange')
            else:
                colors.append('green')
        
        y_pos = np.arange(len(zones))
        bars = ax.barh(y_pos, stress_values, color=colors, alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(zones, fontsize=9)
        ax.set_xlabel('Stress Ratio', fontsize=10)
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Balanced (1.0)')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, stress_values)):
            width = bar.get_width()
            ax.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}', ha='left', va='center', fontsize=8)
        
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_defender_burden_pie(self, ax, results):
        """Plot defender burden distribution"""
        ax.set_title('Defender Burden Distribution', fontsize=12, fontweight='bold')
        
        # Group defenders by burden level
        low_burden = sum(1 for d in results['defender_burdens'].values() if d['burden_score'] < 0.3)
        med_burden = sum(1 for d in results['defender_burdens'].values() if 0.3 <= d['burden_score'] < 0.7)
        high_burden = sum(1 for d in results['defender_burdens'].values() if d['burden_score'] >= 0.7)
        
        sizes = [low_burden, med_burden, high_burden]
        labels = [f'Low (<0.3)\n{low_burden} defenders', 
                 f'Medium (0.3-0.7)\n{med_burden} defenders',
                 f'High (≥0.7)\n{high_burden} defenders']
        colors = ['lightgreen', 'yellow', 'salmon']
        
        # Filter out zero values
        filtered = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
        if filtered:
            sizes, labels, colors = zip(*filtered)
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                               autopct='%1.0f%%', startangle=90)
            
            for text in texts:
                text.set_fontsize(9)
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_fontweight('bold')
        else:
            ax.text(0.5, 0.5, 'No defender data', ha='center', va='center')
