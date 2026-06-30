import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

# Load the data
print("Loading unemployment data...")
df = pd.read_csv('Unemployment in India.csv')

# Data Preprocessing
print("\n" + "="*60)
print("DATA OVERVIEW")
print("="*60)

print(f"\nDataset shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nFirst few rows:")
print(df.head())

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Extract month and year for analysis
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Year_Month'] = df['Date'].dt.strftime('%Y-%m')

print("\n" + "="*60)
print("UNEMPLOYMENT RATE STATISTICS")
print("="*60)

print(f"\nDate range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"\nUnemployment Rate (%) Statistics:")
print(df['Estimated Unemployment Rate (%)'].describe())

# Separate pre-COVID and COVID periods
pre_covid = df[df['Date'] < '2020-03-01']
covid_period = df[df['Date'] >= '2020-03-01']

print(f"\n\nPre-COVID (May 2019 - Feb 2020):")
print(f"Mean Unemployment Rate: {pre_covid['Estimated Unemployment Rate (%)'].mean():.2f}%")
print(f"Std Deviation: {pre_covid['Estimated Unemployment Rate (%)'].std():.2f}%")
print(f"Min: {pre_covid['Estimated Unemployment Rate (%)'].min():.2f}%")
print(f"Max: {pre_covid['Estimated Unemployment Rate (%)'].max():.2f}%")

print(f"\nCOVID Period (March 2020 - June 2020):")
print(f"Mean Unemployment Rate: {covid_period['Estimated Unemployment Rate (%)'].mean():.2f}%")
print(f"Std Deviation: {covid_period['Estimated Unemployment Rate (%)'].std():.2f}%")
print(f"Min: {covid_period['Estimated Unemployment Rate (%)'].min():.2f}%")
print(f"Max: {covid_period['Estimated Unemployment Rate (%)'].max():.2f}%")

print(f"\nIncrease in unemployment during COVID: {covid_period['Estimated Unemployment Rate (%)'].mean() - pre_covid['Estimated Unemployment Rate (%)'].mean():.2f}%")

# Regional Analysis
print("\n" + "="*60)
print("REGIONAL ANALYSIS")
print("="*60)

regional_stats = df.groupby('Region').agg({
    'Estimated Unemployment Rate (%)': ['mean', 'min', 'max', 'std']
}).round(2)

print("\nUnemployment Rate by Region (Overall):")
print(regional_stats)

# Urban vs Rural Analysis
print("\n" + "="*60)
print("URBAN VS RURAL ANALYSIS")
print("="*60)

area_comparison = df.groupby('Area').agg({
    'Estimated Unemployment Rate (%)': ['mean', 'min', 'max', 'std'],
    'Estimated Employed': 'mean'
}).round(2)

print("\nUnemployment Statistics by Area:")
print(area_comparison)

# Pre-COVID vs COVID by Area
print("\n" + "="*60)
print("COVID-19 IMPACT BY AREA")
print("="*60)

for area in df['Area'].unique():
    pre = pre_covid[pre_covid['Area'] == area]['Estimated Unemployment Rate (%)'].mean()
    covid = covid_period[covid_period['Area'] == area]['Estimated Unemployment Rate (%)'].mean()
    impact = covid - pre
    print(f"\n{area} Area:")
    print(f"  Pre-COVID Mean: {pre:.2f}%")
    print(f"  COVID Mean: {covid:.2f}%")
    print(f"  Impact: {impact:+.2f}%")

# Top 5 Most Affected Regions during COVID
print("\n" + "="*60)
print("TOP 5 MOST AFFECTED REGIONS (COVID Period)")
print("="*60)

top_affected = covid_period.groupby('Region')['Estimated Unemployment Rate (%)'].mean().nlargest(5)
print("\n")
for region, rate in top_affected.items():
    print(f"{region}: {rate:.2f}%")

# Time Series Analysis
print("\n" + "="*60)
print("MONTHLY TREND ANALYSIS")
print("="*60)

monthly_trend = df.groupby('Year_Month')['Estimated Unemployment Rate (%)'].mean()
print("\nMonthly Average Unemployment Rate:")
print(monthly_trend)

print("\n" + "="*60)
print("GENERATING VISUALIZATIONS...")
print("="*60)

# 1. Overall Unemployment Trend
fig, ax = plt.subplots(figsize=(14, 6))
monthly_trend.plot(marker='o', linewidth=2, ax=ax, color='#d62728')
ax.axvline(x='2020-03', color='green', linestyle='--', linewidth=2, label='COVID-19 Start (March 2020)')
ax.set_title('Unemployment Rate Trend in India (May 2019 - June 2020)', fontsize=14, fontweight='bold')
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Unemployment Rate (%)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('01_unemployment_trend.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_unemployment_trend.png")
plt.close()

# 2. Pre-COVID vs COVID Comparison
fig, ax = plt.subplots(figsize=(10, 6))
data_comparison = pd.DataFrame({
    'Pre-COVID': [pre_covid['Estimated Unemployment Rate (%)'].mean()],
    'COVID Period': [covid_period['Estimated Unemployment Rate (%)'].mean()]
})
data_comparison.T.plot(kind='bar', ax=ax, legend=False, color=['#1f77b4', '#ff7f0e'])
ax.set_title('Unemployment Rate: Pre-COVID vs COVID Period', fontsize=14, fontweight='bold')
ax.set_ylabel('Unemployment Rate (%)', fontsize=12)
ax.set_xlabel('Period', fontsize=12)
plt.xticks(rotation=0)
for i, v in enumerate([pre_covid['Estimated Unemployment Rate (%)'].mean(), covid_period['Estimated Unemployment Rate (%)'].mean()]):
    ax.text(i, v + 0.2, f'{v:.2f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('02_pre_covid_vs_covid.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_pre_covid_vs_covid.png")
plt.close()

# 3. Urban vs Rural Comparison
fig, ax = plt.subplots(figsize=(10, 6))
area_means = df.groupby('Area')['Estimated Unemployment Rate (%)'].mean()
area_means.plot(kind='bar', ax=ax, color=['#2ca02c', '#d62728'])
ax.set_title('Unemployment Rate: Urban vs Rural Areas', fontsize=14, fontweight='bold')
ax.set_ylabel('Unemployment Rate (%)', fontsize=12)
ax.set_xlabel('Area Type', fontsize=12)
plt.xticks(rotation=0)
for i, v in enumerate(area_means.values):
    ax.text(i, v + 0.2, f'{v:.2f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('03_urban_vs_rural.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_urban_vs_rural.png")
plt.close()

# 4. Top 10 Most Affected Regions (COVID Period)
fig, ax = plt.subplots(figsize=(12, 6))
top_10_regions = covid_period.groupby('Region')['Estimated Unemployment Rate (%)'].mean().nlargest(10)
top_10_regions.plot(kind='barh', ax=ax, color='#9467bd')
ax.set_title('Top 10 Most Affected Regions During COVID-19', fontsize=14, fontweight='bold')
ax.set_xlabel('Average Unemployment Rate (%)', fontsize=12)
plt.tight_layout()
plt.savefig('04_top_10_affected_regions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_top_10_affected_regions.png")
plt.close()

# 5. Regional Variation (Box Plot)
fig, ax = plt.subplots(figsize=(14, 8))
df_sorted = df.sort_values('Region')
sns.boxplot(data=df_sorted, x='Region', y='Estimated Unemployment Rate (%)', ax=ax, palette='Set2')
ax.set_title('Unemployment Rate Distribution by Region', fontsize=14, fontweight='bold')
ax.set_ylabel('Unemployment Rate (%)', fontsize=12)
ax.set_xlabel('Region', fontsize=12)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('05_regional_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_regional_distribution.png")
plt.close()

# 6. COVID Impact by Region and Area
fig, ax = plt.subplots(figsize=(14, 8))
impact_data = []
for region in df['Region'].unique():
    for area in df['Area'].unique():
        subset = df[(df['Region'] == region) & (df['Area'] == area)]
        if len(subset) > 0:
            pre = pre_covid[(pre_covid['Region'] == region) & (pre_covid['Area'] == area)]['Estimated Unemployment Rate (%)'].mean()
            covid = covid_period[(covid_period['Region'] == region) & (covid_period['Area'] == area)]['Estimated Unemployment Rate (%)'].mean()
            if not np.isnan(pre) and not np.isnan(covid):
                impact_data.append({
                    'Region': region,
                    'Area': area,
                    'Pre-COVID': pre,
                    'COVID': covid,
                    'Impact': covid - pre
                })

impact_df = pd.DataFrame(impact_data)
impact_df_pivot = impact_df.pivot_table(values='Impact', index='Region', columns='Area')
sns.heatmap(impact_df_pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'Impact (%)'})
ax.set_title('COVID-19 Impact on Unemployment Rate by Region and Area', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('06_covid_impact_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_covid_impact_heatmap.png")
plt.close()

# 7. Labour Participation Rate Analysis
fig, ax = plt.subplots(figsize=(12, 6))
monthly_participation = df.groupby('Year_Month')['Estimated Labour Participation Rate (%)'].mean()
monthly_participation.plot(marker='s', linewidth=2, ax=ax, color='#17becf')
ax.axvline(x='2020-03', color='green', linestyle='--', linewidth=2, label='COVID-19 Start')
ax.set_title('Labour Participation Rate Trend in India', fontsize=14, fontweight='bold')
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Labour Participation Rate (%)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('07_labour_participation_trend.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 07_labour_participation_trend.png")
plt.close()

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print("\nAll visualizations have been saved to the current folder.")
print("\nKey Findings:")
print(f"  • Average unemployment increased by {covid_period['Estimated Unemployment Rate (%)'].mean() - pre_covid['Estimated Unemployment Rate (%)'].mean():.2f}% during COVID")
print(f"  • Pre-COVID average: {pre_covid['Estimated Unemployment Rate (%)'].mean():.2f}%")
print(f"  • COVID period average: {covid_period['Estimated Unemployment Rate (%)'].mean():.2f}%")
print(f"  • Most affected region: {top_affected.index[0]} ({top_affected.values[0]:.2f}%)")
print("="*60)
