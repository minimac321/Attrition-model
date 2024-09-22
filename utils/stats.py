import pandas as pd
from scipy import stats


"""
Interpreting f_oneway (ANOVA):
ANOVA tests whether there are statistically significant differences between the means of two or more
groups. In your case, you are comparing two groups: Attrition = 1 (Yes) and Attrition = 0 (No).

P-Value Interpretation:
- High p-value (e.g., p > 0.05): No significant difference between the means of the groups.
    - Example: If the p-value for Age is 0.12, it suggests there’s no significant difference in 
    age between employees who left (Attrition = 1) and those who stayed (Attrition = 0).
- Low p-value (e.g., p <= 0.05): Statistically significant difference between the means.
    - Example: If the p-value for Salary is 0.03, it suggests there is a significant difference 
    in salary between employees who left and those who stayed.

Interpreting chi2_contingency (Chi-Square Test):
The Chi-Square test evaluates whether there’s a statistically significant association between two 
categorical variables. In your case, you’re checking if there's an association between Attrition 
and other categorical features (e.g., Department).

P-Value Interpretation:
- High p-value (e.g., p > 0.05): No significant association between the variables.
    - Example: If the p-value for Department is 0.15, it suggests there's no significant relationship
    between department and whether an employee left or stayed.
- Low p-value (e.g., p <= 0.05): Statistically significant association between the variables.

Example: If the p-value for Gender is 0.02, it suggests that gender has a significant association with attrition, meaning gender may influence whether employees leave or stay.
"""


def run_statistical_tests(df, target):
    """
    Perform ANOVA for numeric features and Chi-Square test for categorical features
    against the target variable.

    :param df: DataFrame containing the features and target variable.
    :param target: The target variable (e.g., 'Attrition') for comparison.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    print("ANOVA Test (Numeric Features):")
    for col in numeric_cols:
        if col == target:  # Skip the target column itself
            continue
        one = df[col][df[target] == 1].values  # Attrition = 1
        zero = df[col][df[target] == 0].values  # Attrition = 0
        statistic, p_value = stats.f_oneway(one, zero)
        print(f"P-value for correlation between {target} and {col}: {round(p_value, 4)}")

    print("\nChi-Square Test (Categorical Features):")
    for col in categorical_cols:
        if col == target:  # Skip the target column itself
            continue
        # Create a contingency table
        contingency_table = pd.crosstab(df[col], df[target])
        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
        print(f"P-value for association between {target} and {col}: {round(p_value, 4)}")
