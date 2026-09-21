
##Prompts for Multivariate EDA summarization.


SYSTEM_INSTRUCTION_MULTIVARIATE = (
    "You are a meticulous senior multivariate data analyst and advanced "
    "feature interpretation expert. You must use only supplied facts and "
    "never fabricate statistics."
)


def get_multivariate_prompt(analysis_text_context: str) -> str:
    """
    Build the multivariate EDA summarization prompt.

    Args:
        analysis_text_context: Pre-calculated multivariate EDA metrics
            extracted from the Excel workbook.

    Returns:
        str: Formatted multivariate EDA summarization prompt.
    """

    return f"""
    You are a meticulous senior multivariate data analyst and advanced feature interaction expert.  
    Below are comprehensive pre-calculated multivariate EDA metrics extracted directly from an Excel workbook,  
    covering Variance Inflation Factors (VIF), Principal Component Analysis (PCA), decision tree splits,  
    multivariate models, feature importances, feature interactions, clustering results, and multivariate outliers. 

    Please provide a comprehensive executive summary that analyzes these results, highlighting: 
    1. Multicollinearity risks and redundancy identified through VIF scores. 
    2. Dimensionality reduction insights, variance explained, and principal component loadings from PCA. 
    3. Non-linear relationships, primary splitting features, and predictive drivers surfaced via decision trees and multivariate models. 
    4. Key feature importances, synergistic feature interactions, and natural groupings or clusters identified in the dataset. 
    5. Multivariate outliers, structural anomalies, and actionable feature engineering or dimensionality management recommendations. 

    Workbook Data: 
    {analysis_text_context} 
    """