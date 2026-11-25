#!/usr/bin/env python3
"""
Generate EDA Visuals for Employee Attrition Analysis.

This script loads the raw HR dataset (data/actual/WA_Fn-UseC_-HR-Employee-
Attrition.csv) via src.data_ingestion, applies the matte earth palette from
src.palette (matching the portfolio site), and generates
3 high-resolution plots saved to assets/:
1. attrition_by_department.png
2. compensation_vs_attrition.png
3. correlation_heatmap.png
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_ingestion import ingest_data
from src.palette import (
    BACKGROUND, TEXT, MUTED_TEXT, GRID, REFERENCE,
    LEFT_COLOR, STAYED_COLOR, DIVERGING_CMAP, apply_theme, text_color_for,
)

# Ensure assets directory exists
ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets'))
os.makedirs(ASSETS_DIR, exist_ok=True)

def set_theme():
    """Applies the shared matte earth rcParams, then this script's font stack (unchanged)."""
    apply_theme()
    plt.rcParams.update({
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
        'font.family': 'sans-serif',
    })

def load_data():
    """Loads the HR attrition dataset via src.data_ingestion."""
    return ingest_data()

def generate_attrition_by_department(df):
    """Generates a stylized horizontal bar chart of Attrition Rate (%) by Department."""
    set_theme()
    
    # Calculate Attrition Rate and Total Count per Department
    dept_stats = df.groupby('Department').agg(
        total=('Attrition', 'count'),
        attrition_count=('Attrition', lambda x: (x == 'Yes').sum())
    ).reset_index()
    
    dept_stats['attrition_rate'] = (dept_stats['attrition_count'] / dept_stats['total']) * 100
    dept_stats = dept_stats.sort_values(by='attrition_rate', ascending=True)
    
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    
    y_positions = np.arange(len(dept_stats))
    bars = ax.barh(
        y_positions, 
        dept_stats['attrition_rate'], 
        height=0.55, 
        color=LEFT_COLOR, 
        edgecolor='none' 
    )
    
    # Remove top/right/left spines for clean minimalist layout
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color(REFERENCE)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(dept_stats['Department'], fontsize=11, fontweight='bold', color=TEXT)
    ax.set_xlabel('Attrition Rate (%)', fontsize=11, fontweight='bold', color=MUTED_TEXT, labelpad=10)
    
    # Custom X-axis bounds & grid
    max_val = dept_stats['attrition_rate'].max()
    ax.set_xlim(0, max_val * 1.55)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter())
    ax.grid(True, axis='x')
    ax.set_axisbelow(True)
    
    # Annotate bars with exact percentages and headcount breakdown
    for bar, rate, count, total in zip(bars, dept_stats['attrition_rate'], dept_stats['attrition_count'], dept_stats['total']):
        width = bar.get_width()
        label_text = f" {rate:.1f}%  ({count:,} left of n={total:,})"
        if total < 100:
            label_text += " (small sample)"
        ax.text(
            width + 0.5, 
            bar.get_y() + bar.get_height() / 2, 
            label_text, 
            va='center', 
            ha='left', 
            fontsize=10, 
            fontweight='bold', 
            color=TEXT
        )
        
    # Title & Subtitle without overlapping
    fig.suptitle('Employee Attrition Rate by Department', fontsize=16, fontweight='bold', color=TEXT, x=0.125, y=0.96, ha='left')
    ax.set_title('Share of employees who left, by department (in this dataset)', fontsize=11, color=MUTED_TEXT, pad=15, loc='left')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.86)
    
    output_path = os.path.join(ASSETS_DIR, 'attrition_by_department.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def generate_compensation_vs_attrition(df):
    """Generates a KDE density plot comparing Monthly Income distribution for Attrition Yes vs No."""
    set_theme()
    
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    
    # KDE Plot for Retained vs Departed
    sns.kdeplot(
        data=df[df['Attrition'] == 'No'],
        x='MonthlyIncome',
        fill=True,
        color=STAYED_COLOR,
        alpha=0.25,
        linewidth=2,
        cut=0,
        label='Stayed (Retained)',
        ax=ax
    )
    
    sns.kdeplot(
        data=df[df['Attrition'] == 'Yes'],
        x='MonthlyIncome',
        fill=True,
        color=LEFT_COLOR,
        alpha=0.25,
        linestyle='--',
        linewidth=2,
        cut=0,
        label='Departed (Attrition)',
        ax=ax
    )
    
    # Spines & Grid
    ax.grid(True)
    ax.set_axisbelow(True)
    
    # Formatting Axes
    ax.set_xlabel('Monthly Income', fontsize=11, fontweight='bold', color=MUTED_TEXT, labelpad=10)
    ax.set_ylabel('Density Distribution', fontsize=11, fontweight='bold', color=MUTED_TEXT, labelpad=10)
    # Raw density values mean nothing to a reader: keep the axis label, hide the tick labels
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x/1000:g}k" if x >= 1000 else f"{x:g}"))
    
    # Calculate medians for annotations
    median_stay = df[df['Attrition'] == 'No']['MonthlyIncome'].median()
    median_leave = df[df['Attrition'] == 'Yes']['MonthlyIncome'].median()
    
    # Vertical line indicators for medians, stopping short of the label rows above them
    ax.axvline(median_stay, ymax=0.91, color=STAYED_COLOR, linestyle=':', linewidth=1.5)
    ax.axvline(median_leave, ymax=0.85, color=LEFT_COLOR, linestyle=':', linewidth=1.5)

    # Headroom above both curves; each label is centred on the end of its own line,
    # on its own row, so neither label crosses a line, a curve or the y-axis.
    y_top = ax.get_ylim()[1] * 1.25
    ax.set_ylim(0, y_top)
    ax.text(median_leave, y_top * 0.86, f'Median Departed: {median_leave:,.0f}', color=LEFT_COLOR, fontsize=9.5, fontweight='bold', ha='center', va='bottom')
    ax.text(median_stay, y_top * 0.92, f'Median Stayed: {median_stay:,.0f}', color=STAYED_COLOR, fontsize=9.5, fontweight='bold', ha='center', va='bottom')

    # Legend
    legend = ax.legend(
        fontsize=10, 
        loc='upper right',
        framealpha=0.9
    )
    legend.get_frame().set_linewidth(0.8)
    
    # Title & Subtitle without overlapping
    fig.suptitle('Monthly Income by Attrition Status', fontsize=16, fontweight='bold', color=TEXT, x=0.125, y=0.96, ha='left')
    ax.set_title('Density of monthly income for employees who stayed vs. left (in this dataset)', fontsize=11, color=MUTED_TEXT, pad=15, loc='left')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.86)
    
    output_path = os.path.join(ASSETS_DIR, 'compensation_vs_attrition.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def generate_correlation_heatmap(df):
    """Generates a sleek correlation matrix heatmap of numeric HR variables."""
    set_theme()
    
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
    
    
    # Create mask for upper triangle for clean matrix view
    mask = np.triu(np.ones_like(renamed_corr, dtype=bool))
    
    sns.heatmap(
        renamed_corr, 
        mask=mask,
        cmap=DIVERGING_CMAP, 
        vmax=1.0, 
        vmin=-0.2, 
        center=0,
        annot=True, 
        fmt=".2f", 
        square=True, 
        linewidths=0.8, 
        linecolor=BACKGROUND,
        cbar_kws={"shrink": .8, "label": "Pearson Correlation Coefficient"},
        annot_kws={"size": 8.5, "weight": "bold"},
        ax=ax
    )
    
    # Annotation colour chosen per cell by luminance of the cell fill
    mesh = ax.collections[0]
    for text in ax.texts:
        text.set_color(text_color_for(mesh.cmap(mesh.norm(float(text.get_text())))))

    # Style colorbar text
    cbar = mesh.colorbar
    cbar.ax.yaxis.label.set_color(MUTED_TEXT)
    cbar.ax.yaxis.label.set_fontsize(10)
    cbar.ax.tick_params(colors=MUTED_TEXT, labelsize=9)
    cbar.outline.set_edgecolor(GRID)
    cbar.outline.set_linewidth(0.8)
    
    # Format labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9.5, fontweight='bold', color=MUTED_TEXT)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9.5, fontweight='bold', color=MUTED_TEXT)
    
    # Title & Subtitle without overlapping
    fig.suptitle('Correlation Matrix of Numeric HR Features', fontsize=16, fontweight='bold', color=TEXT, x=0.125, y=0.96, ha='left')
    ax.set_title('Pearson correlations between 12 numeric features; Attrition is not included', fontsize=11, color=MUTED_TEXT, pad=15, loc='left')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    output_path = os.path.join(ASSETS_DIR, 'correlation_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def write_eda_stats(df, output_path=None):
    """
    Writes the EDA figures quoted in the README to assets/eda_stats.json:
    attrition rate by department, median monthly income (left vs stayed),
    and attrition rate by OverTime. Computed on all rows; no plots are drawn.
    """
    if output_path is None:
        output_path = os.path.join(ASSETS_DIR, 'eda_stats.json')
    left = df['Attrition'] == 'Yes'

    def group_rates(column):
        out = {}
        for key, grp in df.groupby(column):
            n_left = int((grp['Attrition'] == 'Yes').sum())
            out[str(key)] = {
                'left': n_left,
                'total': int(len(grp)),
                'rate_pct': round(100 * n_left / len(grp), 1),
            }
        return out

    stats = {
        'rows': int(len(df)),
        'attrition_by_department': group_rates('Department'),
        'median_monthly_income': {
            'left': float(df.loc[left, 'MonthlyIncome'].median()),
            'stayed': float(df.loc[~left, 'MonthlyIncome'].median()),
        },
        'attrition_by_overtime': group_rates('OverTime'),
    }
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
        f.write('\n')
    print(f"Saved: {output_path}")
    return stats

def main():
    print("=" * 60)
    print("Generating High-Resolution EDA Visualizations")
    print("=" * 60)
    
    df = load_data()
    
    print("\n1. Generating Attrition by Department plot...")
    generate_attrition_by_department(df)
    
    print("\n2. Generating Compensation vs. Attrition plot...")
    generate_compensation_vs_attrition(df)
    
    print("\n3. Generating Correlation Heatmap plot...")
    generate_correlation_heatmap(df)
    
    print("\n4. Writing EDA stats...")
    write_eda_stats(df)

    print("\n" + "=" * 60)
    print("All visualizations successfully generated and saved to assets/!")
    print("=" * 60)

if __name__ == '__main__':
    main()
