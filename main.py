import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

final_measurements_delta = pd.read_csv('data/pchip_interpolated_data_with_delta.csv')
reach_centroids_with_Elev = pd.read_csv('data/reach_centroids_with_Elev.csv')
gage_to_wells_df = pd.read_csv('data/gage_well_mapping_revised.csv')
well_reach_relationships = pd.read_csv('data/wells_with_distances.csv')

def filter_wells_by_reach_elevation_with_distribution(
    gage_to_wells_df,
    well_reach_relationships,
    final_measurements_delta,
    reach_centroids_with_Elev,
    elevation_buffer_meters=5,
    delta_bins=[-float('inf'), -20, -10, -5, 0, 5, 10, 20, float('inf')],
    plot=True,
    save_plot_path=None,
    save_output_prefix="output"
):
    tqdm.write("🔄 Step 1: Merging gage-well and well-reach relationships...")
    merged = pd.merge(
        gage_to_wells_df,
        well_reach_relationships[['Well_ID', 'nearest_reach_id', 'lat_dec', 'long_dec', 'Aquifer_Name']],
        left_on='well_id',
        right_on='Well_ID',
        how='inner'
    )

    tqdm.write("🔄 Step 2: Merging with reach elevation data...")
    merged = pd.merge(
        merged,
        reach_centroids_with_Elev[['Reach_ID', 'Avg_GSE']],
        left_on='nearest_reach_id',
        right_on='Reach_ID',
        how='left'
    )

    tqdm.write("🔄 Step 3: Converting WTE to meters...")
    measurements = final_measurements_delta.copy()
    measurements['wte_meters'] = measurements['WTE'] * 0.3048

    tqdm.write("🔄 Step 4: Merging all data with measurements...")
    final = pd.merge(
        merged,
        measurements,
        left_on='Well_ID',
        right_on='Well_ID',
        how='inner'
    )

    tqdm.write("🔄 Step 5: Computing delta_elev and filtering based on elevation buffer...")
    final['reach_elevation_meters'] = final['Avg_GSE']
    final['delta_elev'] = final['wte_meters'] - final['reach_elevation_meters']
    filtered = final[final['delta_elev'] >= -elevation_buffer_meters]

    tqdm.write("🔄 Step 6: Binning elevation delta and computing stats...")
    bin_labels = [f"< {delta_bins[1]}"] + \
                 [f"{delta_bins[i]} to {delta_bins[i+1]}" for i in range(1, len(delta_bins)-2)] + \
                 [f">= {delta_bins[-2]}"]
    final['delta_bin'] = pd.cut(final['delta_elev'], bins=delta_bins, labels=bin_labels)

    total_measurements = len(final)
    total_wells = final['Well_ID'].nunique()

    dist_stats = final.groupby('delta_bin', observed=False).agg(
        measurement_count=('Well_ID', 'count'),
        unique_well_count=('Well_ID', pd.Series.nunique)
    ).reset_index()

    dist_stats['measurement_pct'] = (dist_stats['measurement_count'] / total_measurements * 100).round(2)
    dist_stats['well_pct'] = (dist_stats['unique_well_count'] / total_wells * 100).round(2)

    dist_stats.columns = [col.lower() for col in dist_stats.columns]

    tqdm.write("📦 Step 7: Preparing filtered output columns...")
    expected_columns = [
        'Well_ID', 'gage_id', 'latitude', 'longitude', 'Date', 'WTE', 'Delta_WTE',
        'wte_meters', 'reach_elevation_meters', 'delta_elev',
        'lat_dec', 'long_dec', 'nearest_reach_id'
    ]

    # 确保所有列都存在，不存在的用 NaN 占位
    for col in expected_columns:
        if col not in filtered.columns:
            filtered.loc[:, col] = np.nan  # 安全地添加新列

    output = filtered[expected_columns].copy()
    output.columns = [col.lower() for col in output.columns]

    if plot:
        tqdm.write("📊 Step 8: Plotting...")
        fig, ax1 = plt.subplots(figsize=(10, 6))
        x = dist_stats['delta_bin']
        ax1.bar(x, dist_stats['measurement_pct'], width=0.4, label='measurement %', align='center', alpha=0.7)
        ax2 = ax1.twinx()
        ax2.plot(x, dist_stats['well_pct'], color='orange', marker='o', label='well %')
        ax1.set_ylabel('Measurement %')
        ax2.set_ylabel('Well %')
        ax1.set_xlabel('Δ Elevation Bin (m)')
        ax1.set_title('Elevation Difference Distribution: Measurement % and Well %')
        ax1.tick_params(axis='x', rotation=45)
        fig.tight_layout()
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        if save_plot_path:
            plt.savefig(save_plot_path, dpi=300)
            tqdm.write(f"✅ Plot saved to: {save_plot_path}")

        plt.show()

    # Save outputs
    output_csv = f"{save_output_prefix}_filtered_output.csv"
    dist_csv = f"{save_output_prefix}_distribution_output.csv"

    tqdm.write(f"💾 Step 9: Saving filtered data to {output_csv}")
    output.to_csv(output_csv, index=False)

    tqdm.write(f"💾 Step 10: Saving distribution stats to {dist_csv}")
    dist_stats.to_csv(dist_csv, index=False)

    tqdm.write("✅ All done!")
    return output, dist_stats


# ---------------- MAIN --------------------

if __name__ == "__main__":
    from tqdm import tqdm
    tqdm.write("🚀 Loading data...")

    final_measurements_delta = pd.read_csv('data/pchip_interpolated_data_with_delta.csv')
    reach_centroids_with_Elev = pd.read_csv('data/reach_centroids_with_Elev.csv')
    gage_to_wells_df = pd.read_csv('data/gage_well_mapping_revised.csv')
    well_reach_relationships = pd.read_csv('data/wells_with_distances.csv')

    filtered_df, distribution_df = filter_wells_by_reach_elevation_with_distribution(
        gage_to_wells_df,
        well_reach_relationships,
        final_measurements_delta,
        reach_centroids_with_Elev,
        elevation_buffer_meters=5,
        plot=True,
        save_plot_path="elevation_distribution.png",
        save_output_prefix="results"
    )
