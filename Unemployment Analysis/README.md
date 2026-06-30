# Unemployment Rate Analysis in India

## Project Overview
This project analyzes the unemployment rate trends in India from May 2019 to June 2020, with a particular focus on the impact of COVID-19 pandemic on employment statistics.

## Dataset Information
- **Source**: Unemployment in India.csv
- **Date Range**: May 2019 - June 2020
- **Regions**: All Indian states and union territories
- **Area Types**: Rural and Urban
- **Key Metrics**:
  - Estimated Unemployment Rate (%)
  - Estimated Employed (number of people)
  - Estimated Labour Participation Rate (%)

## Project Structure
```
Unemployment Analysis/
├── unemployment_analysis.py      # Main analysis script
├── Unemployment in India.csv     # Dataset
├── README.md                     # Project documentation
└── [Generated Visualizations]
    ├── 01_unemployment_trend.png
    ├── 02_pre_covid_vs_covid.png
    ├── 03_urban_vs_rural.png
    ├── 04_top_10_affected_regions.png
    ├── 05_regional_distribution.png
    ├── 06_covid_impact_heatmap.png
    └── 07_labour_participation_trend.png
```

## Key Findings

### COVID-19 Impact
- **Sharp Increase**: Unemployment rate showed a significant spike starting from March 2020
- **Peak Unemployment**: Several regions experienced unemployment rates exceeding 40% during April-May 2020
- **Regional Variation**: Different states experienced different levels of impact

### Pre-COVID vs COVID Period
- **Pre-COVID Average** (May 2019 - Feb 2020): Stable unemployment rates around 8-10%
- **COVID Period Average** (March 2020 - June 2020): Dramatic increase with average around 15-20%
- **Overall Impact**: An increase of approximately 7-10% in unemployment rate

### Urban vs Rural Analysis
- Urban areas showed higher average unemployment rates
- Rural areas experienced significant spikes during lockdown periods
- Labour force participation declined sharply in both categories during COVID

## How to Run the Analysis

### Requirements
```bash
pip install pandas numpy matplotlib seaborn
```

### Execute the Script
```bash
python unemployment_analysis.py
```

The script will:
1. Load and clean the dataset
2. Generate comprehensive statistical summaries
3. Perform regional and temporal analysis
4. Create 7 different visualizations
5. Save all outputs to the `Unemployment Analysis` folder

## Visualizations Generated

### 1. **Unemployment Trend (01_unemployment_trend.png)**
   - Time series plot showing unemployment rate from May 2019 to June 2020
   - Clearly marks the COVID-19 onset period (March 2020)

### 2. **Pre-COVID vs COVID Comparison (02_pre_covid_vs_covid.png)**
   - Bar chart comparing average unemployment rates
   - Quantifies the pandemic's immediate impact

### 3. **Urban vs Rural Analysis (03_urban_vs_rural.png)**
   - Comparative analysis of urban and rural unemployment rates
   - Shows regional employment dynamics

### 4. **Top 10 Most Affected Regions (04_top_10_affected_regions.png)**
   - Identifies regions with highest unemployment during COVID
   - Highlights regional disparities

### 5. **Regional Distribution Box Plot (05_regional_distribution.png)**
   - Shows variability of unemployment rates across regions
   - Identifies outliers and regional patterns

### 6. **COVID Impact Heatmap (06_covid_impact_heatmap.png)**
   - 2D visualization of impact by region and area type
   - Color-coded intensity of unemployment increase

### 7. **Labour Participation Trend (07_labour_participation_trend.png)**
   - Tracks labour force participation throughout the period
   - Shows economic participation decline during pandemic

## Analysis Insights

### Key Observations
1. **Abrupt Rise in April 2020**: Unemployment peaked in April 2020 across most regions
2. **Gradual Recovery**: Gradual decline in unemployment from May-June 2020 as lockdown restrictions eased
3. **Regional Disparities**: States like Bihar, Jharkhand, and parts of Northeast India showed higher unemployment rates
4. **Labour Market Shock**: Sharp decline in labour participation rate coinciding with unemployment surge

### Data Quality
- Clean dataset with no missing values in key metrics
- Consistent data collection across all regions
- Both rural and urban areas represented

## Technologies Used
- **Python 3.x**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Data visualization

## Conclusions

The analysis clearly demonstrates:
1. **COVID-19 had a severe impact** on India's employment landscape
2. **The impact was not uniform** - different regions and area types were affected differently
3. **Labour market showed resilience** with signs of recovery by June 2020
4. **Urban areas showed higher vulnerability** to unemployment spikes
5. **Policy interventions may have helped** in the gradual recovery observed

## Future Improvements
- Extend analysis to full 2020 and beyond
- Incorporate sectoral employment data
- Include migration patterns during lockdown
- Analyze government intervention effectiveness
- Compare with international unemployment trends

## Author
**Created as part of OASIS INFOBYTE Internship Program**

## License
Open for educational and analytical purposes
