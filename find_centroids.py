import pandas as pd

def calculate_and_store_quadtree_stats(csv_path, output_path):
    df = pd.read_csv(csv_path)
    df['quadtree_10_1000'] = df['quadtree_10_1000'].astype(int)
    
    def safe_agg(x):
        return pd.Series({
            'min_lat': x['latitude'].min(),
            'max_lat': x['latitude'].max(),
            'mean_lat': x['latitude'].mean(),
            'min_lon': x['longitude'].min(),
            'max_lon': x['longitude'].max(),
            'mean_lon': x['longitude'].mean()
        })
    
    grouped = df.groupby('quadtree_10_1000').apply(safe_agg).reset_index()
    grouped = grouped.rename(columns={'quadtree_10_1000': 'cluster_id'})
    
    # Reorder columns to match desired output
    grouped = grouped[['cluster_id', 'min_lat', 'min_lon', 'max_lat', 'max_lon', 'mean_lat', 'mean_lon']]
    
    # Check for inconsistencies
    inconsistencies = grouped[
        (grouped['mean_lat'] < grouped['min_lat']) | 
        (grouped['mean_lat'] > grouped['max_lat']) |
        (grouped['mean_lon'] < grouped['min_lon']) | 
        (grouped['mean_lon'] > grouped['max_lon'])
    ]
    
    if not inconsistencies.empty:
        print(f"Warning: {len(inconsistencies)} clusters have inconsistent min/max/mean values.")
        print(inconsistencies)
    
    # Save as CSV
    grouped.to_csv(output_path, index=False)
    
    print(f"Processed {len(df)} rows into {len(grouped)} unique quadtree cells.")
    print(f"Results saved to {output_path}")

# Usage
root_dir = 'datasets/osv5m'
train_csv = 'train.csv'
csv_path = f"{root_dir}/{train_csv}"
output_path = f"{root_dir}/quadtree_10_1000_updated.csv"

calculate_and_store_quadtree_stats(csv_path, output_path)