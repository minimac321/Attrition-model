from typing import Optional
import pandas as pd
import numpy as np


# Function to add two random fields
def add_random_survey_fields(df: pd.DataFrame, seed_value: int) -> pd.DataFrame:
    np.random.seed(seed_value)  # For reproducibility

    # Adding a 'Work-Life Balance Score' (numeric, 1 to 5)
    df['WorkLifeBalanceScore_dummy'] = np.random.randint(1, 6, df.shape[0])

    # Adding a 'Work-Culture Satisfaction Score' (numeric, 1 to 5)
    df['WorkPlaceSatisfactionScore_dummy'] = np.random.randint(1, 6, df.shape[0])
    
    # Adding 'Employee Engagement Level' (categorical)
    engagement_levels = ['High', 'Medium', 'Low']
    df['EmployeeEngagementLevel_dummy'] = np.random.choice(engagement_levels, df.shape[0], p=[0.4, 0.4, 0.2])
    
    # Adding a 'Training Opportunity' availability (boolean)
    train_opportunities_levels = [True, False]
    df['TrainingOpportunitiesBool_dummy'] = np.random.choice(train_opportunities_levels, df.shape[0], p=[0.4, 0.6])

    fields_added = [
        "WorkLifeBalanceScore_dummy", "WorkPlaceSatisfactionScore_dummy",
        "EmployeeEngagementLevel_dummy", "TrainingOpportunitiesBool_dummy",
    ]
    return df, fields_added



# Feature 1: Years in Current Role to Years at Company Ratio
def years_in_role_to_years_at_company(df: pd.DataFrame) -> pd.DataFrame:
    df['PercentOfYearsInCurrentRole'] = np.where(df['YearsAtCompany'] != 0, 
                                         df['YearsInCurrentRole'] / df['YearsAtCompany'], 
                                         np.nan)
    return df

# Feature 2: Years Since Last Promotion to Years at Company Ratio
def years_since_promotion_to_years_at_company(df: pd.DataFrame) -> pd.DataFrame:
    df['YearsSincePromoOverTotalYearsRatio'] = np.where(
        df['YearsAtCompany'] != 0, 
        df['YearsSinceLastPromotion'] / df['YearsAtCompany'], 
        np.nan
    )
    return df

# Feature 3: Salary Hike Percentage in Relation to Monthly Income
def salary_hike_to_monthly_income(df: pd.DataFrame) -> pd.DataFrame:
    df['SalaryHikeToIncome'] = np.where(
        df['MonthlyIncome'] != 0,
        df['PercentSalaryHike'] / df['MonthlyIncome'].astype(float), 
        np.nan
    )
    return df

# Feature 4: Income as a Percent of Department Average
def income_percent_of_department_avg(df: pd.DataFrame) -> pd.DataFrame:
    df['DeptAvgIncome'] = df.groupby('Department')['MonthlyIncome'].transform('mean')
    df['IncomeToDeptAvg'] = np.where(df['DeptAvgIncome'] != 0, 
                                     df['MonthlyIncome'] / df['DeptAvgIncome'], 
                                     np.nan)
    df.drop(columns=['DeptAvgIncome'], inplace=True)
    return df

# Feature 5: Overtime per Department Ratio
def satisfaction_to_job_role_avg(df: pd.DataFrame) -> pd.DataFrame:
    job_role_mode = df.groupby('JobRole')['JobSatisfaction'].agg(lambda x: x.mode()[0])
    df['JobRoleAvgSatisfaction'] = df['JobRole'].map(job_role_mode)

    def satisfaction_comparison(row):
        if row['JobSatisfaction'] > row['JobRoleAvgSatisfaction']:
            return 'greater'
        elif row['JobSatisfaction'] == row['JobRoleAvgSatisfaction']:
            return 'equal'
        else:
            return 'less'

    df['SatisfactionToJobRoleAvg'] = df.apply(satisfaction_comparison, axis=1)
    if 'JobRoleAvgSatisfaction' in df.columns:
        df.drop(columns=['JobRoleAvgSatisfaction'], inplace=True)

    return df

# Feature 6: Overtime per Department Ratio
def overtime_per_department_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df['OvertimeFlag'] = np.where(df['OverTime'] == 'Yes', 1, 0)
    df['DeptOvertimeRatio'] = df.groupby('Department')['OvertimeFlag'].transform('mean')
    df.drop(columns=['OvertimeFlag'], inplace=True)
    return df

# Feature 7: Distance From Home Grouping with Percentile Cuts
def distance_from_home_grouping(df: pd.DataFrame) -> pd.DataFrame:
    percentiles = df['DistanceFromHome'].quantile([0.25, 0.5, 0.75])
    bins = [0, percentiles[0.25], percentiles[0.5], percentiles[0.75], np.inf]
    labels = ['Close', 'Moderate', 'Far', 'Very Far']
    df['DistanceGroup'] = pd.cut(df['DistanceFromHome'], bins=bins, labels=labels)
    return df

# Feature 8: Age Group
def age_group(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 30, 40, 50, np.inf]
    labels = ['Under 30', '30-40', '40-50', '50+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
    return df

# Feature 9: Marital Status & Gender Interaction
def marital_status_gender_interaction(df: pd.DataFrame) -> pd.DataFrame:
    df['MaritalGenderInteraction'] = df['MaritalStatus'] + '_' + df['Gender']
    return df

# Feature 10: Interaction between OverTime and JobSatisfaction
def overtime_job_satisfaction_interaction(df: pd.DataFrame) -> pd.DataFrame:
    df['OvertimeJobSatisfactionInteraction'] = df['OverTime'] + '_' + df['JobSatisfaction'].astype(str)
    return df

# Feature 11: Interaction between YearsAtCompany and PerformanceRating
def years_company_performance_interaction(df: pd.DataFrame) -> pd.DataFrame:
    df['YearsAtCompanyPerformanceInteraction'] = df['YearsAtCompany'] * df['PerformanceRating'].astype(float)
    return df

# Feature 12: Interaction between PerformanceRating and JobSatisfaction
def performance_rating_job_satisfaction_interaction(df: pd.DataFrame) -> pd.DataFrame:
    df['OvertimeJobSatisfactionInteraction'] = df['PerformanceRating'] + '_' + df['JobSatisfaction'].astype(str)
    return df

# Feature 13: Overtime per Department Ratio
def satisfaction_to_department_avg(df: pd.DataFrame) -> pd.DataFrame:
    department_mode = df.groupby('Department')['JobSatisfaction'].agg(lambda x: x.mode()[0])
    df['DepartmentAvgSatisfaction'] = df['Department'].map(department_mode)

    def satisfaction_comparison(row):
        if row['JobSatisfaction'] > row['DepartmentAvgSatisfaction']:
            return 'greater'
        elif row['JobSatisfaction'] == row['DepartmentAvgSatisfaction']:
            return 'equal'
        else:
            return 'less'

    df['SatisfactionToDepartmentAvg'] = df.apply(satisfaction_comparison, axis=1)
    if 'DepartmentAvgSatisfaction' in df.columns:
        df.drop(columns=['DepartmentAvgSatisfaction'], inplace=True)

    return df

# NaN Checking Function
def check_for_nan(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if df[column].isna().any():
            print(f"Warning: {column} contains NaN values")
    return df


# Safe feature engineering: Wrap feature engineering in try-except to handle unspecified fields
def safe_feature_engineering(feature_func: callable, df: pd.DataFrame, required_columns: Optional[list] = None) -> pd.DataFrame:
    try:
        # Check if required columns exist, skip if not
        if required_columns is None or all(col in df.columns for col in required_columns):
            return feature_func(df)
        else:
            print(f"Skipping {feature_func.__name__}, missing required columns: {required_columns}")
            return df
    except Exception as e:
        print(f"Error applying {feature_func.__name__}: {e}")
        return df
    
    
# Main function to apply all feature engineering steps and handle sparse data
def apply_feature_engineering(df: pd.DataFrame, add_dummy_data: bool =True) -> pd.DataFrame:
    df = df.copy()
    original_cols = set(df.columns)
    
    # Apply each feature engineering step safely (only if columns exist)
    df = safe_feature_engineering(years_in_role_to_years_at_company, df, ['YearsInCurrentRole', 'YearsAtCompany'])
    df = safe_feature_engineering(years_since_promotion_to_years_at_company, df, ['YearsSinceLastPromotion', 'YearsAtCompany'])
    df = safe_feature_engineering(salary_hike_to_monthly_income, df, ['PercentSalaryHike', 'MonthlyIncome'])
    # Remove this feature due to a very high correleation with MonthlyIncome
    # df = safe_feature_engineering(income_percent_of_department_avg, df, ['MonthlyIncome', 'Department'])
    df = safe_feature_engineering(satisfaction_to_job_role_avg, df, ['JobRole', 'JobSatisfaction'])
    df = safe_feature_engineering(satisfaction_to_department_avg, df, ['JobSatisfaction', 'Department'])
    df = safe_feature_engineering(overtime_per_department_ratio, df, ['OverTime', 'Department'])
    df = safe_feature_engineering(distance_from_home_grouping, df, ['DistanceFromHome'])
    df = safe_feature_engineering(age_group, df, ['Age'])
    df = safe_feature_engineering(marital_status_gender_interaction, df, ['MaritalStatus', 'Gender'])
    df = safe_feature_engineering(overtime_job_satisfaction_interaction, df, ['OverTime', 'JobSatisfaction'])
    df = safe_feature_engineering(years_company_performance_interaction, df, ['YearsAtCompany', 'PerformanceRating'])
    df = safe_feature_engineering(performance_rating_job_satisfaction_interaction, df, ['PerformanceRating', 'JobSatisfaction'])
    
    additional_columns = list(set(df.columns) - set(original_cols))
    
    return df, additional_columns



# # Function to impute missing values
# def impute_missing_data(df: pd.DataFrame) -> pd.DataFrame:
#     # Use median imputation for numeric fields and most frequent for categorical
#     numeric_columns = df.select_dtypes(include=[np.number]).columns
#     categorical_columns = df.select_dtypes(include=[object]).columns
    
#     # Impute numeric data with median
#     imputer_numeric = SimpleImputer(strategy='median')
#     df[numeric_columns] = imputer_numeric.fit_transform(df[numeric_columns])
    
#     # Impute categorical data with most frequent
#     imputer_categorical = SimpleImputer(strategy='most_frequent')
#     df[categorical_columns] = imputer_categorical.fit_transform(df[categorical_columns])
    
#     return df


    


# Main function to apply all feature engineering steps and handle sparse data
# def apply_feature_engineering(df: pd.DataFrame, add_dummy_data: bool =True) -> pd.DataFrame:
#     df = df.copy()
    
#     # Handle missing data
#     df = impute_missing_data(df)
    
#     # Apply each feature engineering step safely (only if columns exist)
#     df = safe_feature_engineering(years_in_role_to_years_at_company, df, ['YearsInCurrentRole', 'YearsAtCompany'])
#     df = safe_feature_engineering(years_since_promotion_to_years_at_company, df, ['YearsSinceLastPromotion', 'YearsAtCompany'])
#     df = safe_feature_engineering(salary_hike_to_monthly_income, df, ['PercentSalaryHike', 'JobLevel'])
#     df = safe_feature_engineering(income_percent_of_department_avg, df, ['MonthlyIncome', 'Department'])
#     df = safe_feature_engineering(income_percent_of_jobrole_avg, df, ['MonthlyIncome', 'JobRole'])
#     df = safe_feature_engineering(overtime_per_department_ratio, df, ['OverTime', 'Department'])
#     df = safe_feature_engineering(distance_from_home_grouping, df, ['DistanceFromHome'])
#     df = safe_feature_engineering(age_group, df, ['Age'])
#     df = safe_feature_engineering(marital_status_gender_interaction, df, ['MaritalStatus', 'Gender'])
#     df = safe_feature_engineering(overtime_job_satisfaction_interaction, df, ['OverTime', 'JobSatisfaction'])
#     df = safe_feature_engineering(years_company_performance_interaction, df, ['YearsAtCompany', 'PerformanceRating'])
    
#     return df

