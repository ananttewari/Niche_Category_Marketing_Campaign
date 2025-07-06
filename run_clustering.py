import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def run_customer_clustering():
    """Run customer clustering and save results to CSV"""
    
    print("Loading data...")
    # Load the marketing campaign data
    data = pd.read_csv("marketing_campaign.csv", sep="\t")
    
    print("Feature engineering and cleaning...")
    # Feature engineering as in the notebook
    if 'Year_Birth' in data.columns:
        data["Age"] = 2021 - data["Year_Birth"]
    data["Spent"] = data["MntWines"] + data["MntFruits"] + data["MntMeatProducts"] + data["MntFishProducts"] + data["MntSweetProducts"] + data["MntGoldProds"]
    data["Living_With"] = data["Marital_Status"].replace({"Married": "Partner", "Together": "Partner", "Absurd": "Alone", "Widow": "Alone", "YOLO": "Alone", "Divorced": "Alone", "Single": "Alone"})
    data["Children"] = data["Kidhome"] + data["Teenhome"]
    data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner": 2}) + data["Children"]
    data["Is_Parent"] = np.where(data.Children > 0, 1, 0)
    data["Education"] = data["Education"].replace({"Basic": "Undergraduate", "2n Cycle": "Undergraduate", "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate"})
    data = data.rename(columns={"MntWines": "Wines", "MntFruits": "Fruits", "MntMeatProducts": "Meat", "MntFishProducts": "Fish", "MntSweetProducts": "Sweets", "MntGoldProds": "Gold"})
    
    # Drop NAs
    data = data.dropna()
    
    # Remove outliers
    if 'Age' in data.columns:
        data = data[data["Age"] < 90]
    if 'Income' in data.columns:
        data = data[data["Income"] < 600000]
    
    # Drop columns as in the notebook
    to_drop = [col for col in ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"] if col in data.columns]
    data = data.drop(to_drop, axis=1)
    cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response']
    data = data.drop([col for col in cols_del if col in data.columns], axis=1)
    
    # Prepare data for clustering
    clustering_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    X = data[clustering_columns].fillna(0)  # Fill missing values with 0
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Applying PCA for dimensionality reduction...")
    # Apply PCA for dimensionality reduction (exactly as in notebook)
    pca = PCA(n_components=3)  # Use 3 components as in notebook
    X_pca = pca.fit_transform(X_scaled)
    
    print("Running Agglomerative Clustering...")
    # Perform Agglomerative Clustering
    AC = AgglomerativeClustering(n_clusters=4)
    clusters = AC.fit_predict(X_pca)
    
    # Add clusters to the original dataframe
    data["Clusters"] = clusters
    
    print("Saving results...")
    # Save the data with clusters
    data.to_csv("marketing_campaign_with_clusters.csv", index=False)
    
    # Print cluster distribution
    cluster_counts = data["Clusters"].value_counts()
    print("\nCluster Distribution:")
    for cluster, count in cluster_counts.items():
        print(f"Cluster {cluster}: {count} customers")
    
    # Identify niche cluster (smallest)
    niche_cluster = cluster_counts.idxmin()
    niche_data = data[data["Clusters"] == niche_cluster]
    print(f"\nNiche Cluster: {niche_cluster}")
    print(f"Niche Cluster Size: {len(niche_data)} customers")
    
    return data

if __name__ == "__main__":
    print("Starting Customer Clustering...")
    clustered_data = run_customer_clustering()
    print("Clustering completed! Data saved to 'marketing_campaign_with_clusters.csv'")