#!/usr/bin/env python3
"""
Generate Custom Dark-Mode Analytics Visuals for Employee Attrition Analysis.

This script loads the raw HR telemetry data (data/actual/WA_Fn-UseC_-HR-Employee-
Attrition.csv) via src.data_ingestion, configures a premium dark-mode
matplotlib/seaborn aesthetic with a copper & teal color palette, and generates
3 high-resolution plots saved to assets/:
1. attrition_by_department.png
2. compensation_vs_attrition.png
3. correlation_heatmap.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_ingestion import ingest_data

# Ensure assets directory exists
ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets'))
os.makedirs(ASSETS_DIR, exist_ok=True)

# Define Dark Mode & Color Palette Constants
BG_COLOR = "#0B0F19"         # Deep Slate / Midnight Navy
CARD_BG = "#111827"          # Slightly lighter panel background
TEXT_COLOR = "#F8FAFC"       # High contrast off-white text
MUTED_TEXT = "#94A3B8"       # Muted slate text
GRID_COLOR = "#1E293B"       # Subtle grid line color

# Brand Palette: Copper & Teal
TEAL_PRIMARY = "#14B8A6"     # Bright Teal (Retained / Positive / Low Risk)
TEAL_SECONDARY = "#0D9488"   # Darker Teal
COPPER_PRIMARY = "#F97316"   # Vibrant Copper / Amber (Attrition / High Risk)
COPPER_SECONDARY = "#C2410C" # Darker Copper

def set_dark_theme():
    """Configures global matplotlib parameters for a premium Dark Mode aesthetic."""
    plt.rcParams.update({
        'figure.facecolor': BG_COLOR,
        'axes.facecolor': BG_COLOR,
        'savefig.facecolor': BG_COLOR,
        'savefig.edgecolor': 'none',
        'text.color': TEXT_COLOR,
        'axes.labelcolor': TEXT_COLOR,
        'xtick.color': MUTED_TEXT,
        'ytick.color': MUTED_TEXT,
        'grid.color': GRID_COLOR,
        'grid.linestyle': '--',
        'grid.alpha': 0.4,
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
        'font.family': 'sans-serif',
        'axes.edgecolor': GRID_COLOR,
        'axes.linewidth': 0.8,
    })

def load_data():
    """Loads the HR attrition dataset via src.data_ingestion."""
    return ingest_data()

def generate_attrition_by_department(df):
    """Generates a stylized horizontal bar chart of Attrition Rate (%) by Department."""
    set_dark_theme()
    
    # Calculate Attrition Rate and Total Count per Department
    dept_stats = df.groupby('Department').agg(
        total=('Attrition', 'count'),
        attrition_count=('Attrition', lambda x: (x == 'Yes').sum())
    ).reset_index()
    
    dept_stats['attrition_rate'] = (dept_stats['attrition_count'] / dept_stats['total']) * 100
    dept_stats = dept_stats.sort_values(by='attrition_rate', ascending=True)
    
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    y_positions = np.arange(len(dept_stats))
    bars = ax.barh(
        y_positions, 
        dept_stats['attrition_rate'], 
        height=0.55, 
        color=COPPER_PRIMARY, 
        edgecolor='none',
        alpha=0.9
    )
    
    # Remove top/right/left spines for clean minimalist layout
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color(GRID_COLOR)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(dept_stats['Department'], fontsize=11, fontweight='bold', color=TEXT_COLOR)
    ax.set_xlabel('Attrition Rate (%)', fontsize=11, fontweight='bold', color=MUTED_TEXT, labelpad=10)
    
    # Custom X-axis bounds & grid
    max_val = dept_stats['attrition_rate'].max()
    ax.set_xlim(0, max_val * 1.18)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter())
    ax.grid(True, axis='x', color=GRID_COLOR, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    # Annotate bars with exact percentages and headcount breakdown
    for bar, rate, count, total in zip(bars, dept_stats['attrition_rate'], dept_stats['attrition_count'], dept_stats['total']):
        width = bar.get_width()
        label_text = f" {rate:.1f}%  ({count:,} / {total:,})"
        ax.text(
            width + 0.5, 
            bar.get_y() + bar.get_height() / 2, 
            label_text, 
            va='center', 
            ha='left', 
            fontsize=10, 
            fontweight='bold', 
            color=COPPER_PRIMARY
        )
        
    # Title & Subtitle without overlapping
    fig.suptitle('Employee Attrition Rate by Department', fontsize=16, fontweight='bold', color=TEXT_COLOR, x=0.125, y=0.96, ha='left')
    ax.set_title('Percentage of departing workforce across organizational units', fontsize=11, color=MUTED_TEXT, pad=15, loc='left')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.86)
    
    output_path = os.path.join(ASSETS_DIR, 'attrition_by_department.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def generate_compensation_vs_attrition(df):
    """Generates a KDE density plot comparing Monthly Income distribution for Attrition Yes vs No."""
    set_dark_theme()
    
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    # KDE Plot for Retained vs Departed
    sns.kdeplot(
        data=df[df['Attrition'] == 'No'],
        x='MonthlyIncome',
        fill=True,
        color=TEAL_PRIMARY,
        alpha=0.35,
        linewidth=2,
        label='Stayed (Retained)',
        ax=ax
    )
    
    sns.kdeplot(
        data=df[df['Attrition'] == 'Yes'],
        x='MonthlyIncome',
        fill=True,
        color=COPPER_PRIMARY,
        alpha=0.45,
        linewidth=2,
        label='Departed (Attrition)',
        ax=ax
    )
    
    # Spines & Grid
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(GRID_COLOR)
        
    ax.grid(True, color=GRID_COLOR, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    # Formatting Axes
    ax.set_xlabel('Monthly Income ($)', fontsize=11, fontweight='bold', color=MUTED_TEXT, labelpad=10)
    ax.set_ylabel('Density Distribution', fontsize=11, fontweight='bold', color=MUTED_TEXT, labelpad=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"${int(x/1000)}k" if x >= 1000 else f"${int(x)}"))
    
    # Calculate medians for annotations
    median_stay = df[df['Attrition'] == 'No']['MonthlyIncome'].median()
    median_leave = df[df['Attrition'] == 'Yes']['MonthlyIncome'].median()
    
    # Vertical line indicators for medians
    ax.axvline(median_stay, color=TEAL_PRIMARY, linestyle=':', linewidth=1.5, alpha=0.8)
    ax.axvline(median_leave, color=COPPER_PRIMARY, linestyle=':', linewidth=1.5, alpha=0.8)
    
    ax.text(median_stay + 1000, ax.get_ylim()[1] * 0.75, f'Median Stayed: ${median_stay:,.0f}', color=TEAL_PRIMARY, fontsize=9.5, fontweight='bold')
    ax.text(median_leave + 1000, ax.get_ylim()[1] * 0.55, f'Median Departed: ${median_leave:,.0f}', color=COPPER_PRIMARY, fontsize=9.5, fontweight='bold')

    # Legend
    legend = ax.legend(
        facecolor=CARD_BG, 
        edgecolor=GRID_COLOR, 
        fontsize=10, 
        labelcolor=TEXT_COLOR, 
        loc='upper right',
        framealpha=0.9
    )
    legend.get_frame().set_linewidth(0.8)
    
    # Title & Subtitle without overlapping
    fig.suptitle('Compensation Density vs. Attrition Risk', fontsize=16, fontweight='bold', color=TEXT_COLOR, x=0.125, y=0.96, ha='left')
    ax.set_title('Monthly income distribution comparison revealing flight risk thresholds', fontsize=11, color=MUTED_TEXT, pad=15, loc='left')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.86)
    
    output_path = os.path.join(ASSETS_DIR, 'compensation_vs_attrition.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def generate_correlation_heatmap(df):
    """Generates a sleek correlation matrix heatmap of numeric HR variables."""
    set_dark_theme()
    
    # Select key numerical features
    num_cols = [
        'Age', 'MonthlyIncome', 'TotalWorkingYears', 
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
        'YearsWithCurrManager', 'PercentSalaryHike', 'DistanceFromHome', 
        'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance'
    ]
    
    # Filter available columns
    available_cols = [c for c in num_cols if c in df.columns]
    corr_matrix = df[available_cols].corr()
    
    # Friendly labels
    label_map = {
        'Age': 'Age',
        'MonthlyIncome': 'Monthly Income',
        'TotalWorkingYears': 'Total Working Yrs',
        'YearsAtCompany': 'Yrs at Company',
        'YearsInCurrentRole': 'Yrs in Role',
        'YearsSinceLastPromotion': 'Yrs Since Promo',
        'YearsWithCurrManager': 'Yrs with Mgr',
        'PercentSalaryHike': '% Salary Hike',
        'DistanceFromHome': 'Distance from Home',
        'JobSatisfaction': 'Job Satisf.',
        'EnvironmentSatisfaction': 'Env. Satisf.',
        'WorkLifeBalance': 'Work-Life Bal.'
    }
    
    renamed_corr = corr_matrix.rename(index=label_map, columns=label_map)
    
    fig, ax = plt.subplots(figsize=(11, 9.5), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    # Custom Diverging Palette (Teal -> Deep Dark Navy -> Copper)
    custom_cmap = sns.diverging_palette(175, 25, s=85, l=55, as_cmap=True, center="dark")
    
    # Create mask for upper triangle for clean matrix view
    mask = np.triu(np.ones_like(renamed_corr, dtype=bool))
    
    sns.heatmap(
        renamed_corr, 
        mask=mask,
        cmap=custom_cmap, 
        vmax=1.0, 
        vmin=-0.2, 
        center=0,
        annot=True, 
        fmt=".2f", 
        square=True, 
        linewidths=0.8, 
        linecolor=BG_COLOR,
        cbar_kws={"shrink": .8, "label": "Pearson Correlation Coefficient"},
        annot_kws={"size": 8.5, "weight": "bold", "color": TEXT_COLOR},
        ax=ax
    )
    
    # Style colorbar text
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(MUTED_TEXT)
    cbar.ax.yaxis.label.set_fontsize(10)
    cbar.ax.tick_params(colors=MUTED_TEXT, labelsize=9)
    cbar.outline.set_edgecolor(GRID_COLOR)
    cbar.outline.set_linewidth(0.8)
    
    # Format labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9.5, fontweight='bold', color=MUTED_TEXT)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9.5, fontweight='bold', color=MUTED_TEXT)
    
    # Title & Subtitle without overlapping
    fig.suptitle('Correlation Matrix of Numerical HR Variables', fontsize=16, fontweight='bold', color=TEXT_COLOR, x=0.125, y=0.96, ha='left')
    ax.set_title('Inter-feature linear relationships across key organizational & telemetry metrics', fontsize=11, color=MUTED_TEXT, pad=15, loc='left')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    output_path = os.path.join(ASSETS_DIR, 'correlation_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    print("=" * 60)
    print("Generating High-Resolution Dark Mode Analytics Visualizations")
    print("=" * 60)
    
    df = load_data()
    
    print("\n1. Generating Attrition by Department plot...")
    generate_attrition_by_department(df)
    
    print("\n2. Generating Compensation vs. Attrition plot...")
    generate_compensation_vs_attrition(df)
    
    print("\n3. Generating Correlation Heatmap plot...")
    generate_correlation_heatmap(df)
    
    print("\n" + "=" * 60)
    print("All visualizations successfully generated and saved to assets/!")
    print("=" * 60)

if __name__ == '__main__':
    main()
