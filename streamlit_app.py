import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import time
import base64

# --- Customer Segmentation Dashboard ---
st.title("Customer Segmentation and Advertisement Generation")

# --- How to Use / Quick Start ---
st.markdown(
    """
    <div style='background-color:#e8f5e9;padding:1.2em 1em 1.2em 1em;border-radius:12px;margin-bottom:1em;'>
    <b>Quick Start Guide</b><br>
    <ul style='margin-bottom:0;'>
      <li><b>1. Explore Segments:</b> Use the buttons below to navigate between customer segmentation, niche analysis, ad generation, and channel strategy.</li>
      <li><b>2. Niche Segment:</b> <span title='A small, unique group of customers identified by clustering.' style='cursor:help;'>Niche Segment</span> is the smallest cluster, ideal for targeted campaigns.</li>
      <li><b>3. Generate Campaigns:</b> In the Ad Campaign section, click <b>Generate Ads</b> to create AI-powered ads and images for your niche audience.</li>
      <li><b>4. Optimize Channels:</b> In Channel Strategy, click <b>Optimize Channel Strategy</b> to get recommended marketing channels for your niche segment.</li>
      <li><b>5. Reset:</b> Click <b>Reset App</b> below to clear selections and start over.</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Reset Button ---
if st.button("🔄 Reset App", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Load data with clusters (run run_clustering.py first if clusters don't exist)
clustered_data_path = "marketing_campaign_with_clusters.csv"
if os.path.exists(clustered_data_path):
    data = pd.read_csv(clustered_data_path)
    st.success("✅ Loaded data with clustering results!")
else:
    data = pd.read_csv(r"marketing_campaign.csv", sep="\t")
    st.warning("⚠️ No clustering results found. Run 'python run_clustering.py' first for full functionality.")

# --- Feature engineering and cleaning (robust to column names) ---
def get_col(data, new, old):
    return new if new in data.columns else old

# Use correct column names for Spent calculation
spent_cols = []
for new, old in zip(["Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold"],
                    ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]):
    spent_cols.append(get_col(data, new, old))

if all(col in data.columns for col in spent_cols):
    data["Spent"] = sum([data[col] for col in spent_cols])

# Age
if 'Age' not in data.columns and 'Year_Birth' in data.columns:
    data["Age"] = 2021 - data["Year_Birth"]
# Living_With
if 'Living_With' not in data.columns and 'Marital_Status' in data.columns:
    data["Living_With"] = data["Marital_Status"].replace({"Married": "Partner", "Together": "Partner", "Absurd": "Alone", "Widow": "Alone", "YOLO": "Alone", "Divorced": "Alone", "Single": "Alone"})
# Children
if 'Children' not in data.columns and 'Kidhome' in data.columns and 'Teenhome' in data.columns:
    data["Children"] = data["Kidhome"] + data["Teenhome"]
# Family_Size
if 'Family_Size' not in data.columns and 'Living_With' in data.columns and 'Children' in data.columns:
    data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner": 2}) + data["Children"]
# Is_Parent
if 'Is_Parent' not in data.columns and 'Children' in data.columns:
    data["Is_Parent"] = np.where(data.Children > 0, 1, 0)
# Education grouping
if 'Education' in data.columns:
    data["Education"] = data["Education"].replace({"Basic": "Undergraduate", "2n Cycle": "Undergraduate", "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate"})
# Rename columns if not already renamed
rename_dict = {"MntWines": "Wines", "MntFruits": "Fruits", "MntMeatProducts": "Meat", "MntFishProducts": "Fish", "MntSweetProducts": "Sweets", "MntGoldProds": "Gold"}
for old, new in rename_dict.items():
    if old in data.columns and new not in data.columns:
        data = data.rename(columns={old: new})
# Drop NAs
if data.isnull().values.any():
    data = data.dropna()
# Remove outliers
if 'Age' in data.columns:
    data = data[data["Age"] < 90]
if 'Income' in data.columns:
    data = data[data["Income"] < 600000]

# Section Selection Buttons
st.header("Explore Customer Segmentation & Campaign Strategy")
col1, col2, col3, col4 = st.columns(4)

with col1:
    section_1 = st.button("📊 Customer Segmentation Overview", use_container_width=True)
with col2:
    section_2 = st.button("🎯 Niche Category Analysis", use_container_width=True)
with col3:
    section_3 = st.button("📢 Integrated Ad Campaign", use_container_width=True)
with col4:
    section_4 = st.button("📈 Channel Strategy Optimization", use_container_width=True)

st.markdown("---")

# Initialize session state for section selection
if 'selected_section' not in st.session_state:
    st.session_state.selected_section = 1

# Update section based on button clicks
if section_1:
    st.session_state.selected_section = 1
elif section_2:
    st.session_state.selected_section = 2
elif section_3:
    st.session_state.selected_section = 3
elif section_4:
    st.session_state.selected_section = 4

# --- Section 1: Customer Segmentation Overview ---
if st.session_state.selected_section == 1:
    st.header("1. Customer Segmentation Overview", help="Customer groups identified by clustering based on similar behaviors and attributes.")
    st.markdown(
        """
        **About the Dataset:**
        This dataset contains customer demographics, purchase history, and campaign responses. It is used to segment customers for targeted marketing.<br>
        <span title='A group of customers with similar characteristics, found using clustering algorithms.' style='cursor:help;'>**Clusters** <sup>?</sup></span> represent these groups.
        """,
        unsafe_allow_html=True
    )
    st.subheader("Dataset Preview")
    st.write(f"Number of customers: {data.shape[0]}")
    st.write(f"Number of features: {data.shape[1]}")
    st.dataframe(data.head())
    st.markdown("""
    The table above shows the first few rows of the dataset, giving an idea of the available features such as customer demographics, purchase history, and campaign responses.
    """)

    st.subheader("Correlation Heatmap")
    st.markdown("""
    The correlation heatmap below shows relationships between numerical features. Strong correlations can indicate similar customer behaviors or redundant features.
    """)
    corr = data.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    plt.close()

    # 3D Plot of Clusters (if available)
    if "Clusters" in data.columns:
        st.subheader("3D Plot of Clusters (PCA Space)")
        st.markdown("""
        This 3D scatter plot shows the distribution of customers in reduced PCA space, colored by cluster. It helps visualize how well the clusters are separated.
        """)
        from sklearn.decomposition import PCA
        from mpl_toolkits.mplot3d import Axes3D
        features = data.select_dtypes(include=[np.number]).drop(columns=["Clusters"]).columns
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(data[features])
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=data["Clusters"], cmap='tab10', alpha=0.7)
        ax.set_title("3D PCA Projection of Clusters")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        plt.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)
        plt.close()

    # Cluster Distribution
    if "Clusters" in data.columns:
        st.subheader("Distribution of Clusters")
        st.markdown("""
        The bar chart below shows the number of customers in each cluster. The smallest cluster is often a niche segment for targeted marketing.
        """)
        st.bar_chart(data["Clusters"].value_counts())

    # Clustering Features by Cluster
    st.subheader("Clustering Features by Cluster")
    st.markdown("""
    The following plots show how key features are distributed across clusters. These insights help understand what makes each segment unique.
    """)
    cluster_features = [
        "Income", "Kidhome", "Teenhome", "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth", "Spent", "Age", "Children", "Family_Size", "Is_Parent"
    ]
    for feature in cluster_features:
        if feature in data.columns and "Clusters" in data.columns:
            st.write(f"#### {feature} by Cluster")
            fig, ax = plt.subplots(figsize=(8, 5))
            if feature == "Is_Parent":
                sns.countplot(x="Clusters", hue=feature, data=data, palette="Set2", ax=ax)
                ax.set_ylabel("Count")
            else:
                sns.violinplot(x="Clusters", y=feature, data=data, palette="Set2", ax=ax)
                ax.set_ylabel(feature)
            ax.set_title(f"{feature} by Cluster")
            st.pyplot(fig)
            plt.close()
            st.markdown("---")
    st.markdown("""
    **Insights:**
    - Features like Income, Spent, and Age often show clear differences between clusters, revealing high-value or niche segments.
    - Features like NumWebPurchases or NumWebVisitsMonth can indicate digital engagement, useful for campaign targeting.
    - The Is_Parent feature can help identify family-oriented segments.
    """)

# --- Section 2: Niche Category Data ---
elif st.session_state.selected_section == 2:
    st.header("2. Niche Category (Smallest Cluster) Data", help="The smallest, most unique customer group—ideal for micro-campaigns.")

    if "Clusters" in data.columns:
        cluster_counts = data["Clusters"].value_counts()
        niche_cluster = cluster_counts.idxmin()
        niche_data = data[data["Clusters"] == niche_cluster]
        st.write(f"Niche cluster label: {niche_cluster}")
        st.write(f"Number of members in niche cluster: {len(niche_data)}")
        st.dataframe(niche_data.head())
        # Show summary stats
        st.subheader("Niche Cluster Summary Stats")
        st.write(niche_data.describe())

        # Feature distribution in niche cluster
        st.subheader("Feature Distributions in Niche Cluster")
        dist_features = [
            "Income", "Spent", "Age", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth", "Children", "Family_Size"
        ]
        for feature in dist_features:
            if feature in niche_data.columns:
                st.write(f"**{feature} Distribution in Niche Cluster**")
                fig, ax = plt.subplots(figsize=(7, 3))
                sns.histplot(niche_data[feature], kde=True, ax=ax, color="#9F8A78")
                ax.set_title(f"{feature} Distribution (Niche Cluster)")
                st.pyplot(fig)
                plt.close()

        # Compare niche cluster to others
        st.subheader("Niche vs. Other Clusters Comparison")
        compare_features = ["Income", "Spent", "Age"]
        for feature in compare_features:
            if feature in data.columns:
                st.write(f"**{feature}: Niche vs. Others**")
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.violinplot(x="Clusters", y=feature, data=data, palette="Set2", ax=ax)
                ax.axvline(niche_cluster, color='red', linestyle='--', label='Niche Cluster')
                ax.set_title(f"{feature} by Cluster (Niche Highlighted)")
                st.pyplot(fig)
                plt.close()

        # Categorical breakdowns
        st.subheader("Categorical Feature Breakdown in Niche Cluster")
        cat_features = ["Education", "Living_With", "Is_Parent"]
        for feature in cat_features:
            if feature in niche_data.columns:
                st.write(f"**{feature} Breakdown**")
                fig, ax = plt.subplots(figsize=(6, 3))
                niche_data[feature].value_counts().plot(kind='bar', ax=ax, color="#B9C0C9")
                ax.set_title(f"{feature} Distribution (Niche Cluster)")
                st.pyplot(fig)
                plt.close()

        # Insights
        st.markdown("""
        **Niche Cluster Insights:**
        - The above plots show the unique characteristics of the niche segment.
        - Compare the niche cluster's feature distributions to the rest to spot what makes this group special.
        - Categorical breakdowns (e.g., Education, Living_With, Is_Parent) help identify the lifestyle and preferences of this segment.
        """)
    else:
        st.warning("No 'Clusters' column found. Please run clustering in the notebook and export results.")

# --- Section 3: Integrated Ad Campaign ---
elif st.session_state.selected_section == 3:
    st.header("3. Integrated Ad: Copy & Image")

    st.subheader("Niche Segment Ad Generator")
    st.markdown("""
    _Click the button below to generate three unique, AI-powered ads (copy + image) tailored for your niche customer segment. Each click gives you a fresh set of creative ads!_
    """)
    
    # Add a button to trigger ad generation
    if 'ad_generation' not in st.session_state:
        st.session_state.ad_generation = False
    if st.button("Generate Ads for Niche Category", use_container_width=True):
        st.session_state.ad_generation = True
        st.session_state.generated_ads = None  # Reset previous ads

    # Only show ads if button was clicked
    if st.session_state.ad_generation:
        import requests
        import random
        import time
        import uuid
        
        # --- GenAI pipeline config ---
        BRAND_NAME = "EcoElevate"  # Cool brand name
        
        # API Configuration - using Streamlit secrets
        try:
            GROQ_API_KEY = st.secrets["api_keys"]["GROQ_API_KEY"]
        except KeyError:
            st.error("⚠️ API keys not found. Please configure your secrets in `.streamlit/secrets.toml`")
            st.info("Add your API keys to `.streamlit/secrets.toml` file with the following format:")
            st.code("""
[api_keys]
GROQ_API_KEY = "your_groq_api_key_here"
CLOUDFLARE_API_KEY = "your_cloudflare_api_key_here"
CLOUDFLARE_ACCOUNT_ID = "your_cloudflare_account_id_here"
            """)
            GROQ_API_KEY = None
        
        # Ad creative templates
        ad_headline = "Elevate Your Workspace, Not Your Carbon Footprint"
        ad_subheading = "Discover the Future of Sustainable Office Supplies"
        ad_body_template = (
            f"As a professional, you know that making a statement is crucial. But did you know that your office supplies can make a significant impact on the environment? "
            f"Traditional office supplies often come wrapped in plastic, made from non-renewable resources, and end up in landfills. It's time to rethink your workspace.\n\n"
            f"Introducing {BRAND_NAME}, the premier destination for modern, eco-friendly office supplies. Our commitment to sustainability is reflected in every item we produce. "
            f"From recycled materials to minimalist design, {BRAND_NAME} helps you work smarter, greener, and with style. Make the switch today and join the movement for a cleaner tomorrow!"
        )
        
        def build_prompts(i):
            # Add a unique seed/angle for each ad
            seed = random.randint(1000, 9999) + int(time.time()) + i
            ad_prompt = (
                f"Headline: {ad_headline}\n"
                f"Subheading: {ad_subheading}\n"
                f"Body: {ad_body_template}\n"
                f"Unique Angle: {angles[i % len(angles)]} (Seed: {seed})\n"
                f"Write this as a single, visually engaging ad for {BRAND_NAME}."
            )
            image_prompt = (
                f"A modern, eco-friendly office workspace for professionals, minimalist design, recycled materials, lush plants, sunlight, photorealistic, vibrant. (Seed: {seed})"
            )
            return ad_prompt, image_prompt, seed
        
        ad_creatives = []
        angles = [
            "Minimalist recycled essentials for the modern professional.",
            "Brighten your desk with plant-based, zero-waste supplies.",
            "Sleek, sustainable tools for eco-conscious achievers."
        ]
        # --- Generate a single image for all ads ---
        image_prompt = None
        image_path = None
        image_fetched = False
        seed = random.randint(1000, 9999) + int(time.time())
        image_prompt = (
            f"A modern, eco-friendly office workspace for professionals, minimalist design, recycled materials, lush plants, sunlight, photorealistic, vibrant. (Seed: {seed})"
        )
        # Try Cloudflare Workers AI for image generation (text-to-image, SDXL)
        def generate_image_cloudflare(image_prompt, seed):
            api_key = st.secrets["api_keys"]["CLOUDFLARE_API_KEY"]
            account_id = st.secrets["api_keys"]["CLOUDFLARE_ACCOUNT_ID"]
            API_BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/"
            headers = {"Authorization": f"Bearer {api_key}"}
            payload = {
                "prompt": image_prompt,
                "seed": seed,
                "height": 1024,
                "width": 1024,
                "num_steps": 20,
                "guidance": 7.5
            }
            model_slug = "@cf/stabilityai/stable-diffusion-xl-base-1.0"
            api_url = f"{API_BASE_URL}{model_slug}"
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            content_type = response.headers.get("Content-Type", "")
            if content_type.startswith("image/"):
                image_path = f"niche_campaign_image_{uuid.uuid4().hex[:8]}.png"
                with open(image_path, "wb") as f:
                    f.write(response.content)
                return image_path
            try:
                result = response.json()
            except Exception:
                return None
            if not result:
                return None
            if "result" in result and isinstance(result["result"], str):
                img_data = base64.b64decode(result["result"])
                image_path = f"niche_campaign_image_{uuid.uuid4().hex[:8]}.png"
                with open(image_path, "wb") as f:
                    f.write(img_data)
                return image_path
            return None
        try:
            image_path = generate_image_cloudflare(image_prompt, seed)
            image_fetched = image_path is not None
            if not image_fetched:
                # Use fallback image if AI generation fails
                fallback_image = "Epsilon Hackathon Default Image.png"
                if os.path.exists(fallback_image):
                    image_path = fallback_image
                    image_fetched = True
                else:
                    st.warning("Cloudflare API did not return a valid image and fallback image is missing.")
        except Exception as e:
            st.error(f"Image generation failed: {e}")
            fallback_image = "Epsilon Hackathon Default Image.png"
            if os.path.exists(fallback_image):
                image_path = fallback_image
                image_fetched = True
            else:
                image_path = None
                image_fetched = False
        if not image_fetched:
            image_path = None
        # --- Generate 3 ad copies, all using the same image ---
        for i in range(3):
            ad_prompt, _, seed_i = build_prompts(i)
            # --- Generate ad copy (Groq/OpenAI) ---
            ad_text = None
            if GROQ_API_KEY:
                try:
                    api_url = "https://api.groq.com/openai/v1/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": "llama3-8b-8192",
                        "messages": [
                            {"role": "system", "content": f"You are a creative ad copywriter for {BRAND_NAME}. Write a single, visually engaging ad. Use the following HTML template for all ads: <b style='font-size:1.5em;color:#2e7d32;'>HEADLINE</b><br><span style='font-size:1.1em;color:#388e3c;'><i>SUBHEADING</i></span><br><br>BODY. Do not use asterisks or markdown. No bullet points, no lists, no image section. Make each ad copy unique and do not repeat the same content."},
                            {"role": "user", "content": ad_prompt + f'\nMake this ad copy different from the previous ones.'}
                        ],
                        "max_tokens": 256,
                        "temperature": 0.98 + 0.01 * i  # More variety
                    }
                    response = requests.post(api_url, headers=headers, json=payload, timeout=30)
                    result = response.json()
                    if "choices" in result and result["choices"]:
                        ad_text = result["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    ad_text = None
            if not ad_text or '**' in ad_text:
                # Fallback: add more uniqueness and remove all asterisks
                unique_tail = f"<span style='color:#1976d2;'>{angles[i % len(angles)].replace('*','')}</span>"
                ad_text = (
                    f"<b style='font-size:1.5em;color:#2e7d32;'>{ad_headline} {i+1}</b><br>"
                    f"<span style='font-size:1.1em;color:#388e3c;'><i>{ad_subheading} (Version {i+1})</i></span><br><br>"
                    f"{ad_body_template}<br><br>{unique_tail}"
                )
            ad_text = ad_text.replace('**','').replace('*','')
            ad_creatives.append((ad_text, image_path))
            time.sleep(1)
        st.session_state.generated_ads = ad_creatives
        st.success("Generated 3 unique, visually engaging ads for the niche segment!")

    # Display generated ads if available
    if st.session_state.ad_generation and st.session_state.generated_ads:
        # Display the image only once, above all ad copies
        first_image_path = st.session_state.generated_ads[0][1] if st.session_state.generated_ads[0][1] and os.path.exists(st.session_state.generated_ads[0][1]) else None
        if first_image_path:
            st.image(Image.open(first_image_path), caption="Niche Campaign Image", use_container_width=True)
        for i, (ad, _) in enumerate(st.session_state.generated_ads, 1):
            st.markdown("---")
            st.markdown(f"<div style='background:#f4f7f3;padding:2em 1em 2em 1em;border-radius:18px;box-shadow:0 2px 8px #e0e0e0;margin-bottom:2em;'>" + ad.replace("\n", "<br>") + "</div>", unsafe_allow_html=True)
    elif not st.session_state.ad_generation:
        st.info("Click the button above to generate AI-powered ads for your niche customer segment!")
# --- Section 4: Channel Strategy Optimization ---
if st.session_state.selected_section == 4:
    st.header("4. Channel Strategy Optimization for Niche Category")
    st.markdown("""
    _Optimize your marketing channel mix for the niche customer segment. This section uses a rules-based approach to recommend the best channels based on the segment's demographic and behavioral profile._
    """)
    if "Clusters" in data.columns:
        cluster_counts = data["Clusters"].value_counts()
        niche_cluster = cluster_counts.idxmin()
        niche_data = data[data["Clusters"] == niche_cluster]
        st.write(f"Niche cluster label: {niche_cluster}")
        st.write(f"Number of members in niche cluster: {len(niche_data)}")
        # Summarize key features
        def summarize_niche_features(niche_data):
            summary = {}
            for col in niche_data.select_dtypes(include='number').columns:
                if col == 'Clusters':
                    continue
                vals = niche_data[col]
                summary[col] = int(vals.median()) if vals.nunique() > 1 else int(vals.iloc[0])
            for col in [c for c in niche_data.columns if niche_data[c].dtype == object or str(niche_data[c].dtype).startswith('category')]:
                mode_val = niche_data[col].mode().iloc[0] if not niche_data[col].mode().empty else None
                summary[col] = mode_val
            if 'Is_Parent' in niche_data:
                summary['Is_Parent'] = 'Yes' if niche_data['Is_Parent'].mode().iloc[0] == 1 else 'No'
            return summary
        niche_summary = summarize_niche_features(niche_data)
        st.subheader("Niche Segment Profile Summary")
        st.json(niche_summary)
        # Channel recommendation logic
        def recommend_channels(niche_summary):
            age = int(niche_summary.get('Age', 0))
            income = int(niche_summary.get('Income', 0))
            education = str(niche_summary.get('Education', '')).lower()
            parent = niche_summary.get('Is_Parent', 'No') == 'Yes'
            spent = int(niche_summary.get('Spent', 0))
            family_size = int(niche_summary.get('Family_Size', 1))
            channel_scores = {
                'Instagram': 0,
                'TikTok': 0,
                'Email': 0,
                'LinkedIn': 0,
                'Facebook': 0,
                'YouTube': 0
            }
            if age < 30:
                channel_scores['TikTok'] += 2
                channel_scores['Instagram'] += 2
                channel_scores['YouTube'] += 1
            elif age < 45:
                channel_scores['Instagram'] += 2
                channel_scores['LinkedIn'] += 2
                channel_scores['Email'] += 1
                channel_scores['YouTube'] += 1
            else:
                channel_scores['Facebook'] += 2
                channel_scores['Email'] += 2
                channel_scores['LinkedIn'] += 1
            if income > 80000:
                channel_scores['LinkedIn'] += 2
                channel_scores['Email'] += 1
            if 'postgraduate' in education or 'graduate' in education:
                channel_scores['LinkedIn'] += 2
                channel_scores['Email'] += 1
            if parent or family_size > 2:
                channel_scores['Facebook'] += 1
                channel_scores['YouTube'] += 1
            if spent > 1000:
                channel_scores['Instagram'] += 1
                channel_scores['LinkedIn'] += 1
            return channel_scores
        if st.button("Optimize Channel Strategy", use_container_width=True):
            channel_scores = recommend_channels(niche_summary)
            st.subheader("Recommended Channel Mix (Bar Chart)")
            st.bar_chart(channel_scores)
            sorted_channels = sorted(channel_scores.items(), key=lambda x: x[1], reverse=True)
            st.markdown("**Top Recommended Channels:** " + ", ".join([ch for ch, score in sorted_channels if score > 0]))
            st.markdown("""
            **Rationale:**
            - Recommendations are based on age, income, education, parenthood, family size, and spending patterns.
            """)
            st.markdown("---")
            st.markdown("**Channel Score Details:**")
            for ch, score in sorted_channels:
                st.write(f"- {ch}: {score}")
    else:
        st.warning("No 'Clusters' column found. Please run clustering in the notebook and export results.")

st.markdown("---")
st.write("Built with Streamlit for rapid MVP demo. Data and results are synced with the main notebook.")
