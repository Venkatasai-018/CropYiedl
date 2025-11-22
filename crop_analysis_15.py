"""
Random Forest Model - 15 Crop Analysis
=====================================

Analysis of parameter ranges and recommendations for 15 different crops
based on the trained Random Forest model dataset.
"""

import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('Crop Recommendation dataset.csv')

print("🌾 RANDOM FOREST MODEL - 15 CROP ANALYSIS")
print("=" * 60)

# Select 15 most diverse crops for analysis
selected_crops = [
    'rice', 'maize', 'wheat', 'cotton', 'banana', 'apple', 'mango', 
    'grapes', 'orange', 'pomegranate', 'chickpea', 'kidneybeans', 
    'lentil', 'coffee', 'coconut'
]

# Filter crops that exist in dataset (since wheat might not be present)
available_crops = df['label'].unique()
selected_crops = [crop for crop in selected_crops if crop in available_crops]

# If we don't have 15, add more from available crops
while len(selected_crops) < 15 and len(selected_crops) < len(available_crops):
    for crop in available_crops:
        if crop not in selected_crops:
            selected_crops.append(crop)
            if len(selected_crops) == 15:
                break

print(f"Analyzing {len(selected_crops)} crops from the Random Forest model:")
print("-" * 60)

# Analyze each crop
crop_analysis = {}

for i, crop in enumerate(selected_crops, 1):
    crop_data = df[df['label'] == crop]
    
    analysis = {
        'N_range': f"{crop_data['N'].min():.0f} - {crop_data['N'].max():.0f} kg/ha",
        'N_avg': f"{crop_data['N'].mean():.1f} kg/ha",
        'P_range': f"{crop_data['P'].min():.0f} - {crop_data['P'].max():.0f} kg/ha", 
        'P_avg': f"{crop_data['P'].mean():.1f} kg/ha",
        'K_range': f"{crop_data['K'].min():.0f} - {crop_data['K'].max():.0f} kg/ha",
        'K_avg': f"{crop_data['K'].mean():.1f} kg/ha",
        'temp_range': f"{crop_data['temperature'].min():.1f} - {crop_data['temperature'].max():.1f} °C",
        'temp_avg': f"{crop_data['temperature'].mean():.1f} °C",
        'humidity_range': f"{crop_data['humidity'].min():.1f} - {crop_data['humidity'].max():.1f} %",
        'humidity_avg': f"{crop_data['humidity'].mean():.1f} %",
        'ph_range': f"{crop_data['ph'].min():.1f} - {crop_data['ph'].max():.1f}",
        'ph_avg': f"{crop_data['ph'].mean():.1f}",
        'rainfall_range': f"{crop_data['rainfall'].min():.0f} - {crop_data['rainfall'].max():.0f} mm",
        'rainfall_avg': f"{crop_data['rainfall'].mean():.1f} mm",
        'sample_count': len(crop_data)
    }
    
    crop_analysis[crop] = analysis
    
    print(f"{i:2d}. {crop.upper()}")
    print(f"    Nitrogen (N)    : {analysis['N_range']} (avg: {analysis['N_avg']})")
    print(f"    Phosphorus (P)  : {analysis['P_range']} (avg: {analysis['P_avg']})")
    print(f"    Potassium (K)   : {analysis['K_range']} (avg: {analysis['K_avg']})")
    print(f"    Temperature     : {analysis['temp_range']} (avg: {analysis['temp_avg']})")
    print(f"    Humidity        : {analysis['humidity_range']} (avg: {analysis['humidity_avg']})")
    print(f"    pH Level        : {analysis['ph_range']} (avg: {analysis['ph_avg']})")
    print(f"    Rainfall        : {analysis['rainfall_range']} (avg: {analysis['rainfall_avg']})")
    print(f"    Samples         : {analysis['sample_count']}")
    print()

print("\n" + "=" * 60)
print("OPTIMAL PARAMETER RANGES SUMMARY")
print("=" * 60)

# Create summary table
summary_data = []
for crop in selected_crops:
    crop_data = df[df['label'] == crop]
    summary_data.append({
        'Crop': crop.title(),
        'N_avg': round(crop_data['N'].mean(), 1),
        'P_avg': round(crop_data['P'].mean(), 1),
        'K_avg': round(crop_data['K'].mean(), 1),
        'Temp_avg': round(crop_data['temperature'].mean(), 1),
        'Humidity_avg': round(crop_data['humidity'].mean(), 1),
        'pH_avg': round(crop_data['ph'].mean(), 1),
        'Rainfall_avg': round(crop_data['rainfall'].mean(), 1)
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print(f"\n" + "=" * 60)
print("CROP CATEGORIES BY NUTRIENT REQUIREMENTS")
print("=" * 60)

# Categorize crops by nutrient requirements
high_n_crops = summary_df[summary_df['N_avg'] > 80]['Crop'].tolist()
medium_n_crops = summary_df[(summary_df['N_avg'] >= 40) & (summary_df['N_avg'] <= 80)]['Crop'].tolist()
low_n_crops = summary_df[summary_df['N_avg'] < 40]['Crop'].tolist()

high_rainfall_crops = summary_df[summary_df['Rainfall_avg'] > 150]['Crop'].tolist()
low_rainfall_crops = summary_df[summary_df['Rainfall_avg'] < 100]['Crop'].tolist()

warm_climate_crops = summary_df[summary_df['Temp_avg'] > 25]['Crop'].tolist()
cool_climate_crops = summary_df[summary_df['Temp_avg'] < 20]['Crop'].tolist()

print("HIGH NITROGEN REQUIREMENT (>80 kg/ha):")
print(f"  {', '.join(high_n_crops)}")
print(f"\nMEDIUM NITROGEN REQUIREMENT (40-80 kg/ha):")
print(f"  {', '.join(medium_n_crops)}")
print(f"\nLOW NITROGEN REQUIREMENT (<40 kg/ha):")
print(f"  {', '.join(low_n_crops)}")

print(f"\nHIGH RAINFALL CROPS (>150mm):")
print(f"  {', '.join(high_rainfall_crops)}")
print(f"\nLOW RAINFALL CROPS (<100mm):")
print(f"  {', '.join(low_rainfall_crops)}")

print(f"\nWARM CLIMATE CROPS (>25°C):")
print(f"  {', '.join(warm_climate_crops)}")
print(f"\nCOOL CLIMATE CROPS (<20°C):")
print(f"  {', '.join(cool_climate_crops)}")

print("\n" + "=" * 60)
print("SAMPLE PREDICTION INPUTS FOR EACH CROP")
print("=" * 60)

print("Use these optimal parameter combinations for testing the Random Forest model:\n")

for i, crop in enumerate(selected_crops, 1):
    crop_data = df[df['label'] == crop]
    
    # Get median values as optimal parameters
    optimal_params = {
        'N': int(crop_data['N'].median()),
        'P': int(crop_data['P'].median()),
        'K': int(crop_data['K'].median()),
        'temperature': round(crop_data['temperature'].median(), 1),
        'humidity': round(crop_data['humidity'].median(), 1),
        'ph': round(crop_data['ph'].median(), 1),
        'rainfall': round(crop_data['rainfall'].median(), 1)
    }
    
    print(f"{i:2d}. {crop.upper()} - Optimal Parameters:")
    print(f"    {optimal_params}")

print(f"\n" + "=" * 60)
print("FEATURE IMPORTANCE INSIGHTS")
print("=" * 60)

# Calculate correlation of each feature with different crops
feature_importance_insights = {}
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

for feature in features:
    crop_feature_avg = []
    for crop in selected_crops:
        crop_data = df[df['label'] == crop]
        crop_feature_avg.append(crop_data[feature].mean())
    
    # Find crops with highest and lowest values for each feature
    max_idx = np.argmax(crop_feature_avg)
    min_idx = np.argmin(crop_feature_avg)
    
    feature_importance_insights[feature] = {
        'highest_crop': selected_crops[max_idx],
        'highest_value': round(crop_feature_avg[max_idx], 1),
        'lowest_crop': selected_crops[min_idx],
        'lowest_value': round(crop_feature_avg[min_idx], 1),
        'range': round(crop_feature_avg[max_idx] - crop_feature_avg[min_idx], 1)
    }

for feature, insight in feature_importance_insights.items():
    print(f"{feature.upper()}:")
    print(f"  Highest: {insight['highest_crop']} ({insight['highest_value']})")
    print(f"  Lowest:  {insight['lowest_crop']} ({insight['lowest_value']})")
    print(f"  Range:   {insight['range']}")
    print()

print("=" * 60)
print("RANDOM FOREST MODEL ANALYSIS COMPLETE!")
print("=" * 60)