import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from scipy.stats import skew, kurtosis
from statsmodels.stats.anova import anova_lm
from scipy.stats import zscore


st.set_page_config(
    layout="wide",
    page_title="Pet-Sight",
    page_icon="🐾",
)

# Load your cleaned CSV or query DB
df = pd.read_csv("data/cleaned_pet_proactive_mock_data.csv")

# Ensure visit_date is datetime for time series
df['visit_date'] = pd.to_datetime(df['visit_date'])


# -----------------------------------------------------------------
# helper functions for layout
def spacer(n=1):
    import streamlit as st
    st.markdown("<br>" * n, unsafe_allow_html=True)


#  helper functions for analysis
def eta2_one_way(formula):
    model = ols(formula, data=df).fit()
    aov = anova_lm(model, typ=3)  # Type III SS
    factor = formula.split('~')[1].strip()  # get factor name
    sum_sq_factor = aov.loc[factor, 'sum_sq']
    sum_sq_error = aov.loc['Residual', 'sum_sq']
    eta2 = sum_sq_factor / (sum_sq_factor + sum_sq_error)
    return eta2, aov


# Partial eta-squared helper
def partial_eta2(term):
    ss_term = aov_two.loc[term, 'sum_sq']
    ss_error = aov_two.loc['Residual', 'sum_sq']
    return ss_term / (ss_term + ss_error)


# for correlation analysis
def corr_with_pvals(df, method="pearson"):
    """
    Compute correlation matrix and p-value matrix for numeric dataframe.  

    Args:
        df (pd.DataFrame): numeric dataframe
        method (str): 'pearson' or 'spearman'

    Returns:
        corr_df (pd.DataFrame): correlation matrix
        pval_df (pd.DataFrame): matrix of p-values
    """
    cols = df.columns
    corr_df = df.corr(method=method)
    pval_df = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for i in cols:
        for j in cols:
            if i == j:
                pval_df.loc[i, j] = np.nan
            else:
                if method == "pearson":
                    r, p = stats.pearsonr(df[i], df[j])
                elif method == "spearman":
                    r, p = stats.spearmanr(df[i], df[j])
                else:
                    raise ValueError("method must be 'pearson' or 'spearman'")
                pval_df.loc[i, j] = p

    return corr_df, pval_df

# -----------------------------------------------------------------


# Sidebar for top-level sections
st.sidebar.title("PetSight Dashboard")
section = st.sidebar.radio(
    "Choose Analysis Section",
    [
        "Cost & Insurance",
        "Patients & Demographics",
        "Conditions & Treatments",
        "Visits & Utilization"
    ]
)


# ====================== COST & INSURANCE ======================
if section == "Cost & Insurance":
    st.title("Cost & Insurance Analysis")

    # --- Tabs for main sub-sections ---
    tabs = st.tabs(["Overview / Plots", "Insurance Analysis", "Core Descriptive Analysis (Cost)"])

    # ================== OVERVIEW / PLOTS TAB ==================
    with tabs[0]:
        st.header("Cost Analysis Overview")

        # Metric cards (global / baseline)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Cost (£)", f"{df['cost_gbp'].mean():.2f}")
        col2.metric("Avg Covered (£)", f"{df['covered_gbp'].mean():.2f}")
        col3.metric("Avg Out-of-Pocket (£)", f"{df['out_of_pocket_gbp'].mean():.2f}")
        col4.metric("% Insured", f"{df['insured'].mean()*100:.1f}%")

        st.markdown("---")
        spacer(2)

        # --- 1. Average Cost by Species ---
        fig, ax = plt.subplots(figsize=(8, 5))
        avg_cost = df.groupby("species")["cost_gbp"].mean().sort_values()
        sns.barplot(x=avg_cost.index, y=avg_cost.values, ax=ax, palette="viridis")
        ax.set_title("Average Cost by Species")
        ax.set_ylabel("Avg Cost (£)")
        st.pyplot(fig)

        st.markdown("""
        Average costs increase slightly from Cat to Bird, with Bird having the highest average cost among the four species shown.  
        """)
        st.markdown("---")
        spacer(1)

        # --- 2. Average Cost by Treatment ---
        fig, ax = plt.subplots(figsize=(8, 5))
        avg_treat = df.groupby("treatment")["cost_gbp"].mean().sort_values()
        sns.barplot(y=avg_treat.index, x=avg_treat.values, ax=ax, palette="magma")
        ax.set_title("Average Cost by Treatment")
        ax.set_xlabel("Avg Cost (£)")
        st.pyplot(fig)

        st.markdown("""
        Treatment costs are highest for Check-up, vaccination, and medication, with dental cleaning being the lowest among the listed treatments.  
        """)
        st.markdown("---")
        spacer(1)

        # --- 3. Average Cost by Condition ---
        fig, ax = plt.subplots(figsize=(8, 5))
        avg_cond = df.groupby("condition")["cost_gbp"].mean().sort_values(ascending=False)
        sns.barplot(x=avg_cond.index, y=avg_cond.values, ax=ax, palette="plasma")
        ax.set_title("Average Cost by Condition")
        ax.set_ylabel("Avg Cost (£)")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

        st.markdown("""
        Costs decline progressively from Obesity to Ear infection, showing the lowest average cost for the last condition (ear infection) and the highest for obesity and hip dysplasia.  
        """)
        st.markdown("---")
        spacer(1)

        # --- 4. Monthly Average Cost ---
        df["month"] = df["visit_date"].dt.to_period("M").astype(str)
        monthly = df.groupby("month")["cost_gbp"].mean().reset_index()

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(x="month", y="cost_gbp", data=monthly, marker="o", ax=ax)
        ax.set_title("Monthly Average Cost (All Species)")
        ax.set_ylabel("Avg Cost (£)")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

        st.markdown("""
        The costs fluctuate over the year with notable peaks around mid-year (July) and a sharp dip in February and September, indicating seasonal cost variation.  
        """)
        st.markdown("---")
        spacer(1)

        # ================== INSURANCE ANALYSIS TAB ==================
        with tabs[1]:
            st.header("Insurance Overview")

            insured_pct = df['insured'].mean() * 100
            avg_coverage = df[df['insured']]['coverage_rate'].mean()
            avg_covered = df['covered_gbp'].mean()
            avg_out_of_pocket = df['out_of_pocket_gbp'].mean()

            # Metric cards
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Coverage Rate", f"{avg_coverage:.2f}")
            col2.metric("Avg Covered (£)", f"{avg_covered:.2f}")
            col3.metric("Avg Out-of-Pocket (£)", f"{avg_out_of_pocket:.2f}")
            col4.metric("Insured Pets (%)", f"{insured_pct:.1f}%")

            st.markdown("---")
            spacer(2)

            # ---- Coverage Rate by Plan ----
            fig, ax = plt.subplots(figsize=(8,5))
            plan_avg = df[df['insured']].groupby("insurance_plan")['coverage_rate'].mean().sort_values()
            sns.barplot(x=plan_avg.index, y=plan_avg.values, ax=ax, palette="coolwarm")
            ax.set_title("Avg Coverage Rate by Insurance Plan")
            ax.set_ylabel("Coverage Rate")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
            st.markdown("""
            Coverage rate increases with plan generosity from Accident-only to Lifetime  
            - Accident-only plan covers about 60% on average; the more comprehensive plans reach around 78-85% coverage  
            - Time-limited plan and Maximum benefit are mid-to-high coverage options, implying better protection for more types of claims  
            - Lifetime plan offers the highest average coverage, suggesting strong protection against long-term or chronic costs  

            ➡️ For clients prioritizing payout risk protection, higher-tier plans (maximum benefit or lifetime) reduce out-of-pocket exposure  
            ➡️ If cost-conscious, the Accident-only or Time-limited plans still provide substantial coverage but with lower protection  
            """)
            st.markdown("---")
            spacer(1)

            # ---- Out-of-Pocket by Species ----
            fig, ax = plt.subplots(figsize=(8,5))
            species_oop = df.groupby("species")["out_of_pocket_gbp"].mean().sort_values()
            sns.barplot(x=species_oop.index, y=species_oop.values, ax=ax, palette="mako")
            ax.set_title("Avg Out-of-Pocket by Species")
            ax.set_ylabel("Avg Out-of-Pocket (£)")
            st.pyplot(fig)
            st.markdown("""
            Out-of-pocket exposure differs by species  
            - Bird shows the highest out-of-pocket amount among the four shown, followed by Rabbit, Dog, and Cat  
            - This may reflect differences in typical veterinary care costs, treatment frequency, or species-specific procedures  

            ➡️ Pet owners with higher-cost species (e.g., birds) may benefit more from insurance plans with lower co-pays or higher coverage limits   
            """)
            st.markdown("---")
            spacer(1)

            # ---- Covered vs Out-of-Pocket Distribution ----
            fig, ax = plt.subplots(figsize=(8,5))
            sns.boxplot(data=df[['covered_gbp', 'out_of_pocket_gbp']], palette="Set2", ax=ax)
            ax.set_title("Covered vs Out-of-Pocket Distribution")
            ax.set_ylabel("£ Amount")
            st.pyplot(fig)
            st.markdown("""
            Distribution of costs between covered amounts and out-of-pocket amounts  
            - Most cases cluster with relatively low covered amounts, but there is a long tail of higher costs where out-of-pocket is substantial  
            - The box plots show a wide spread for out-of-pocket costs, with some extreme points outside the main group (occasional large bills not fully covered)  

            ➡️ Clients should consider plans that reduce the likelihood or magnitude of high out-of-pocket bills, not just average coverage  
            ➡️ There is risk of catastrophic expenses (visible as upper outliers), should find ways to hedge this risk (e.g., higher coverage plans or annual deductible caps)  
            """)
            st.markdown("---")
            spacer(1)

    # ================== CORE DESCRIPTIVE ANALYSIS TAB ==================
    with tabs[2]:
        st.header("Core Cost Analysis (Descriptive)")
        spacer(3)
        st.markdown("---")
        spacer(1)
        # --- Cost Distribution ---
        st.subheader("Cost Distribution 📈")
        st.markdown("**Cost Distribution: Histogram and KDE of Cost**")
        # log transform for more symmetric views
        df["log_cost"] = np.log1p(df["cost_gbp"])  # log(1+x) to avoid log(0)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df["cost_gbp"], bins=50, kde=True, ax=ax)
        ax.set_title("Distribution of Costs (£)")
        ax.set_xlabel("Cost (£)")
        st.pyplot(fig)  # Streamlit-friendly plotting

        # Skewness & Kurtosis
        skew_ = stats.skew(df["cost_gbp"], nan_policy="omit")
        kurt = stats.kurtosis(df["cost_gbp"], nan_policy="omit")
        st.markdown(f"**Skewness:** {skew_:.2f}  |  **Kurtosis:** {kurt:.2f}")

        # --- Key Summary / Report ---
        st.markdown("""
        **Most costs cluster around £200, with a tail reaching up to about £700.**
        **The peak sits near £200-£210.**
        **The distribution shows a right skew and is slightly flatter than normal (platykurtic).**

        ---

        **Key Numbers:**  
        - Mode: £200-£210
        - Min: £0
        - Max: £700
        - Mean: £208
        - Median: £209
        - Std dev: £129
        - 25th percentile: £106
        - 75th percentile: £293
        - Skewness: 0.45 (positive)
        - Kurtosis: -0.17 (platykurtic)
        """)

        # --- Analysis Report ---
        st.markdown("""
        ---
        #### Analysis

        **Core Metrics:**
        - N = 1,017
        - Center: Mean £208.18; Median £209.67 (median ≈ mean)
        - Spread: SD £129.38; IQR £187.01; CV 0.62
        - Range: £0.14 to £672.91
        - Percentiles: 5th £12.03; 95th £434.49
        - Outliers: 9 (per IQR rule)

        **Shape and Tails:**
        - Skewness 0.446; Kurtosis -0.165
        - Interpretation: mild to moderate right skew; unimodal with a not-too-sharp peak

        **Tail Focus:**
        - 95th percentile £434.49
        - Tail is notable but not extreme relative to max (£672.91)

        **Implications:**
        - Non-normality suggests using robust summaries (median, IQR) or transform-based/GLM approaches
        ---
        """)

        # --- Variance Decomposition ---
        spacer(2)
        st.subheader("Variance Decomposition 📊")
        st.markdown("**One-way ANOVA**")
        # One-way ANOVA
        eta2_species, aov_species = eta2_one_way("cost_gbp ~ C(species)")
        eta2_condition, aov_condition = eta2_one_way("cost_gbp ~ C(condition)")
        eta2_treatment, aov_treatment = eta2_one_way("cost_gbp ~ C(treatment)")

        # Display Eta²
        st.markdown(f"- **Species**: ($\\eta^2$) Eta² = {eta2_species:.4f}")
        st.markdown(f"- **Condition**: ($\\eta^2$) Eta² = {eta2_condition:.4f}")
        st.markdown(f"- **Treatment**: ($\\eta^2$) Eta² = {eta2_treatment:.4f}")
        spacer(2)

        # Display ANOVA tables
        st.markdown("**ANOVA table (Species)**")
        st.table(aov_species)
        spacer(1)

        st.markdown("**ANOVA table (Condition)**")
        st.table(aov_condition)
        spacer(1)

        st.markdown("**ANOVA table (Treatment)**")
        st.table(aov_treatment)
        st.markdown("---")

        spacer(1)
        #  -----------------------------------------------------------
        st.markdown("**Two-way ANOVA with Interaction**")
        model_two = ols("cost_gbp ~ C(species) * C(condition)", data=df).fit()
        aov_two = anova_lm(model_two, typ=3)

        eta2_species = partial_eta2("C(species)")
        eta2_condition = partial_eta2("C(condition)")
        eta2_interaction = partial_eta2("C(species):C(condition)")

        # Display Eta^2
        st.markdown(f"- **Partial ($\\eta^2$) Eta² (Species)** = {eta2_species:.4f}")
        st.markdown(f"- **Partial ($\\eta^2$) Eta² (Condition)** = {eta2_condition:.4f}")
        st.markdown(f"- **Partial ($\\eta^2$) Eta² (Interaction)** = {eta2_interaction:.4f}")

        spacer(1)
        # Two-way ANOVA table
        st.markdown("**Two-way ANOVA Table**")
        # Round for readability
        st.table(aov_two.round(4))

        spacer(1)
        st.markdown("---")
        st.markdown("#### Report")
        # report
        st.markdown("""

        **The cost story isn't driven by a single factor.**
        The most influential driver is the treatment, but the big surprise is the interaction:
        costs differ most when species and condition meet, not when either acts alone.

        **One Way**
        - Treatment:
            - $\\eta^2 \\approx 0.041$
            - $p \\approx 1.19 \\times 10^{-6}$
        - Condition:
            - $\\eta^2 \\approx 0.034$
            - $p \\approx 1.17 \\times 10^{-4}$
        - Species:
            - $\\eta^2 \\approx 0.0033$
            - $p \\approx 0.340$
        - Treatments and Conditions are statistically significant drivers of cost; Species is not. Practical impact is modest ($\\eta^2$ around 0.03-0.04)

        ---

        **Two Way**
        - Interaction partial (species $\\times$ condition)
            - $\\eta^2 \\approx 0.104$
            - $p \\approx 9.02 \\times 10^{-11}$
            - This is significant and the largest effect
        - Main effects (partial $\\eta^2$):
            - species $\\approx 0.0024$
            - condition $\\approx 0.0053$
        - *Interpretation*:
            - The interaction is the primary driver among the three terms; main effects are small when the interaction is included.
            - The interplay between species and condition drives most of the variance.

        **Advice for decision-making**
        - Focus on the interaction: costs vary meaningfully by species within each condition.
        - Single-factor views (species alone or condition alone) are much less informative for budgeting or policy.
        ---
        """)

        # --- Correlation Structure ---
        st.subheader("Correlation Structure 📊")
        st.markdown("Correlation of Cost (GBP), Covered, Paid out of Pocket, Insured")
        cols = ["cost_gbp", "covered_gbp", "out_of_pocket_gbp", "insured"]
        df_corr = df[cols].copy()
        df_corr["insured"] = df_corr["insured"].astype(int)

        pearson_corr, pearson_p = corr_with_pvals(df_corr, method="pearson")
        spearman_corr, spearman_p = corr_with_pvals(df_corr, method="spearman")

        st.markdown("**Pearson Correlations**")
        st.table(pearson_corr.round(3))
        st.markdown("**Spearman Correlations**")
        st.table(spearman_corr.round(3))
        st.markdown("---")

        # ---- Analysis report ----
        st.markdown("""
        **Correlation Structure Report**

        - Pearson (linear)
            -  cost_gbp $\\leftrightarrow$ out_of_pocket_gbp: $r \\approx 0.778$, $p \\approx 2.11 \\times 10^{-207}$
            -  cost_gbp $\\leftrightarrow$ covered_gbp: $r \\approx 0.297$, $p \\approx 4.19 \\times 10^{-22}$
            -  cost_gbp $\\leftrightarrow$ insured: $r \\approx -0.025$, $p \\approx 0.433$
            -  covered_gbp $\\leftrightarrow$ insured: $r \\approx 0.798$, $p \\approx 2.00 \\times 10^{-16}$
            -  out_of_pocket_gbp $\\leftrightarrow$ insured: $r \\approx -0.549$, $p \\approx 2.00 \\times 10^{-16}$

        - Spearman (rank)
            -  cost_gbp $\\leftrightarrow$ out_of_pocket_gbp: $\\rho \\approx 0.768$
            -  covered_gbp $\\leftrightarrow$ insured: $\\rho \\approx 0.975$
            -  cost_gbp $\\leftrightarrow$ covered_gbp: $\\rho \\approx 0.098$

        ---
        - What this means
            -  The total cost signal tracks most strongly with out_of_pocket_gbp; covers a secondary but meaningful link.
            -  Insurance status (insured) does not predict cost on its own.
            -  Coverage and insured move together very tightly; they are a closely linked footprint in the data.

        >  Two visuals will tell the story: a **Pearson heatmap** focused on `cost_gbp` with `out_of_pocket_gbp` and `covered_gbp`, and a **Spearman heatmap** to confirm monotone patterns.
        ---
        """)
        spacer(2)
        st.markdown("### Heatmaps for Pearson and Spearman")
        # # Heatmap: Pearson
        # fig, ax = plt.subplots(figsize=(6, 5))
        # sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        # ax.set_title("Pearson Correlations")
        # plt.tight_layout()
        # st.pyplot(fig)

        # # Heatmap: Spearman
        # fig, ax = plt.subplots(figsize=(6, 5))
        # sns.heatmap(spearman_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        # ax.set_title("Spearman Correlations")
        # plt.tight_layout()
        # st.pyplot(fig)
        spacer(1)
        # Heatmap: Pearson Correlation: ***cost_gbp*** with ***out_of_pocket_gbp*** and ***covered_gbp***
        # st.markdown("Pearson Correlation: ***cost_gbp*** with ***out_of_pocket_gbp*** and ***covered_gbp***")
        cols_focus = ['cost_gbp', 'out_of_pocket_gbp', 'covered_gbp']
        pearson_focus = df[cols_focus].corr(method='pearson')
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(pearson_focus, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title("Pearson correlations: cost_gbp, out_of_pocket_gbp, covered_gbp")
        plt.tight_layout()
        st.pyplot(fig)
        spacer(1)
        st.markdown("""
        Cost_gbp moves strongest with out_of_pocket_gbp ($r \\approx 0,778$; $p \\approx 2,11 \\times 10^{-207}$)  
        and has a moderate link to covered_gbp ($r \\approx 0,297$; $p \\approx 4,19 \\times 10^{-22}$);  
        insurance status shows no linear relation ($r \\approx -0,025$; $p \\approx 0,433$).  
        Spearman confirms the pattern:   ($\\rho \\approx 0,768$ for ***cost_gbp*** vs ***out_of_pocket_gbp***; $\\rho \\approx 0,975$ for ***covered_gbp*** vs ***insured***; 
        $\\rho \\approx 0,098$ for ***cost_gbp*** vs ***covered_gbp***),  
        so we'll use out_of_pocket_gbp as the main cost signal. 
        """)
        st.markdown("---")
        spacer(2)

        # Heatmap: Spearman: confirm monotone patterns
        spearman_focus = df[cols_focus].corr(method='spearman')
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(spearman_focus, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title("Spearman correlations: cost_gbp, out_of_pocket_gbp, covered_gbp")
        plt.tight_layout()
        st.pyplot(fig)
        spacer(1)
        st.markdown("""
        There is a strong monotonic link between cost_gbp and out_of_pocket_gbp $(\\rho \\approx 0.77)$,
        meaning higher total costs tend to coincide with higher out-of-pocket payments.
        There is a moderate negative monotonic link between out_of_pocket_gbp and covered_gbp $(\\rho \\approx -0.52)$ 
        and a very weak monotonic link between cost_gbp and covered_gbp $(\\rho \\approx 0.10)$, 
        so coverage and total cost move together less consistently;
        insurance status itself isn't shown here as a strong predictor.
        """)
        st.markdown("---")
        spacer(2)

        # --- Outlier Analysis ---
        st.subheader("Outlier Analysis ⚠️")
        st.markdown("Top outliers & high-cost clusters")

        # IQR-based outliers
        Q1 = df['cost_gbp'].quantile(0.25)
        Q3 = df['cost_gbp'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers_iqr = df[(df['cost_gbp'] < lower) | (df['cost_gbp'] > upper)]
        st.subheader("IQR Outliers")
        st.table(outliers_iqr[['pet_id', 'cost_gbp', 'condition', 'treatment']])

        # Z-score outliers
        df['cost_z'] = zscore(df['cost_gbp'])
        outliers_z = df[(df['cost_z'].abs() > 3)]
        st.subheader("Z-score Outliers")
        st.table(outliers_z[['pet_id', 'cost_gbp', 'cost_z', 'condition', 'treatment']])

        # Cost clusters
        group_cols = ['species', 'condition', 'treatment']
        summary = df.groupby(group_cols)['cost_gbp'].agg(
            count='size',
            mean='mean',
            median='median',
            max='max'
        ).reset_index()
        high_mean_threshold = summary['mean'].quantile(0.90)
        clusters = summary[(summary['mean'] > high_mean_threshold) & (summary['count'] >= 3)]
        st.subheader("High-Cost Clusters")
        st.table(clusters)

        spacer(1)
        st.markdown("---")
        # report
        st.markdown("""
        **Outlier & Cluster Analysis Report**

        Two patterns stand out in cost_gbp:

        1. nine **IQR outliers** around £580-£590 (examples include P0197 £580,80; P0206 £586,00). 

        2. **Z-score outliers:** A tail is reinforced by three very high values (£603,80; £620,22; £672,91), confirming the upper tail.

        - Four cost clusters drive most costs: 
            - Cat Hip dysplasia Vaccination ≈ £375,24 mean (max ≈ £503),
            - Cat Unknown Medication ≈ £451,57 mean (max ≈ £507),
            - Dog Dental disease Check-up ≈ £364,28 mean (max ≈ £459), 
            - Dog Obesity Vaccination ≈ £466,91 mean (max ≈ £580). 

        The outliers are rare but financially meaningful; clusters highlight high-cost care areas to monitor and review. To manage tail risk, we need to highlight top outliers and tail metrics (90th/95th percentile).
        """)

        spacer(1)
        st.markdown("---")
        # boxplot by species/conditionto visualize clusters
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='species', y='cost_gbp', hue='condition', data=df, ax=ax)
        ax.set_title("Cost distribution by Species and Condition")
        plt.xticks(rotation=45)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        st.pyplot(fig)
        st.markdown("""
        Costs vary widely by both species and condition, with some combos showing higher medians and wider ranges (e.g., certain dog and cat conditions) than others. Within each category there’s substantial spread and several outliers, so budgeting should account for tail risk rather than relying on average costs by species or condition.
        """)
        st.markdown("---")

else:
    st.warning(f"🚧 {section} is under construction! Please check back later.")
    st.stop()

