import numpy as np
import pandas as pd
import scipy as sc
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

from collections import OrderedDict
from itertools import compress
from sklearn import linear_model
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    explained_variance_score
)
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize



def generate_metrics(predicted_probs, labels, threshold=0.5):
    """
    Generate classification and probability-based metrics for binary classification, including Efron's R-squared and Gini score.

    Parameters:
    - predicted_probs (array-like): The predicted probabilities for the positive class.
    - labels (array-like): The true binary labels (0 or 1).
    - threshold (float, optional): The threshold to convert predicted probabilities into binary predictions. Defaults to 0.5.

    Returns:
    - tuple: A tuple containing two dictionaries (one for classification metrics, one for probability-based metrics), 
      Efron's R-squared, Gini score, and the confusion matrix as a pandas dataframe.
    """
    # Convert predicted probabilities to binary predictions
    predicted_labels = (predicted_probs >= threshold).astype(int)
    
    # Initialize dictionaries to store metrics
    classification_metrics = {}
    probability_metrics = {}
    
    # Classification metrics
    classification_metrics['accuracy'] = accuracy_score(labels, predicted_labels)
    classification_metrics['specificity'] = (confusion_matrix(labels, predicted_labels)[0, 0]) / (confusion_matrix(labels, predicted_labels)[0, :].sum())
    classification_metrics['recall'] = recall_score(labels, predicted_labels)
    classification_metrics['precision'] = precision_score(labels, predicted_labels)
    classification_metrics['f1'] = f1_score(labels, predicted_labels)
    
    # Probability-based metrics
    roc_auc = roc_auc_score(labels, predicted_probs)
    probability_metrics['roc_auc'] = roc_auc
    probability_metrics['average_precision'] = average_precision_score(labels, predicted_probs)
    mean_labels = np.mean(labels)
    probability_metrics['efron_r2'] = 1 - (np.sum((labels - predicted_probs) ** 2) / np.sum((labels - mean_labels) ** 2))
    probability_metrics['gini'] = 2 * roc_auc - 1  # Calculate Gini score
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predicted_labels).ravel()
    confusion_matrix_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        columns=['Predicted 0', 'Predicted 1'],
        index=['Actual 0', 'Actual 1']
    )
    
    # Return a tuple with the two dictionaries and the confusion matrix DataFrame
    return probability_metrics, classification_metrics, confusion_matrix_df




def sfa_linear_model(model, dataset, DV, IVs, forced_in=None, get_fitted=True, threshold=0.5):
    '''
    Perform classification on individual independent variables, with optional forced in variables
    :param model: Instantiated model class from https://scikit-learn.org/stable/modules/linear_model.html for binary target variables, that has a 'predict_proba' method to get the predicted probabilities.
    :param dataset: pandas dataframe
    :param DV: name of target variable (dependent variable)
    :param IVs: list of names of independent variables to be evaluated one by one
    :param forced_in: optional list of forced in variables. All variables named here will be forced in
    :param get_fitted: boolean whether to get fitted values
    :return: tuple with 'results' and 'fitted_values' dataframes
    '''

    has_intercept = model.fit_intercept

    # Check inputs and reformat if necessary
    if DV is None:
        raise ValueError("You must include DV")
    if IVs is None:
        raise ValueError("You must include IVs")

    if forced_in is not None:
        if isinstance(forced_in, str):
            forced_in = [forced_in]

    # Set up the result table in Pandas
    col_info = OrderedDict()
    col_info['Variable'] = pd.Series([], dtype='str')
    if forced_in is not None:
        col_info['Forced In'] = pd.Series([], dtype='str')

    col_info['# Obs'] = pd.Series([], dtype='int')
    col_info['# Miss'] = pd.Series([], dtype='int')
    if has_intercept:
        col_info['Intercept'] = pd.Series([], dtype='float')
    col_info['Var Coef'] = pd.Series([], dtype='float')

    if forced_in is not None:
        for forced_var in forced_in:
            col_info[forced_var + " Coef"] = pd.Series([], dtype='float')
    
    metric_names = ['accuracy', 'specificity', 'recall', 'precision', 'f1', 'roc_auc', 'average_precision', 'efron_r2', 'gini', 'True Positive', 'False Positive', 'False Negative', 'True Negative']
    
    for metric in metric_names:
        col_info['Var Coef'] = pd.Series([], dtype='float')
        
    # Create the pandas
    output = pd.DataFrame(col_info)

    # Create Pandas for fitted values
    fitted_values = dataset[[DV]].copy()
    fitted_values.columns = ["Actual"]

    # If there are forced in variables, we run a regression with just the forced in variables
    if forced_in:

        model_dataset = dataset[[DV] + forced_in].dropna()
        kept_index = model_dataset.index.values

        X = model_dataset[forced_in]
        Y = model_dataset[DV]

        results = model.fit(X, Y)

        # Generate outputs
        results_dict = OrderedDict()

        results_dict['Variable'] = "None"
        results_dict['Forced In'] = "|".join(forced_in)
        results_dict['# Obs'] = model_dataset.shape[0]
        results_dict['# Miss'] = len(Y) - model_dataset.shape[0]
        if has_intercept:
            results_dict['Intercept'] = results.intercept_[0]
        results_dict['Var Coef'] = 0

        if forced_in is not None:
            for forced_var in forced_in:
                results_dict[forced_var + " Coef"] = results.coef_[0][results.feature_names_in_.tolist().index(forced_var)]

        predicted = model.predict_proba(X)[:,1]
        forced_in_fitted = predicted.copy()
        
        probability_metrics, classification_metrics, confusion_matrix_df = generate_metrics(predicted_probs=predicted, labels=fitted_values['Actual'], threshold=threshold)
        
        for metric in ['roc_auc', 'average_precision', 'efron_r2', 'gini']:
            results_dict[metric] = probability_metrics[metric]
        
        for metric in ['accuracy', 'specificity', 'recall', 'precision', 'f1']:
            results_dict[metric] = classification_metrics[metric]
            
        results_dict['True Positive'] = confusion_matrix_df.iloc[1,1]
        results_dict['False Positive'] = confusion_matrix_df.iloc[1,0]
        results_dict['False Negative'] = confusion_matrix_df.iloc[0,1]
        results_dict['True Negative'] = confusion_matrix_df.iloc[0,0]
        
        output = pd.concat([output, pd.DataFrame(results_dict, index=[0])]).reset_index(drop=True)

        # Fitted values
        if get_fitted:
            fitted_values["ForcedInVars"] = np.nan
            fitted_values.loc[kept_index, "ForcedInVars"] = predicted

    # Loop through variables
    for IV in IVs:
        print("Working on {}, which is #{} out of {}".format(IV, IVs.index(IV) + 1, len(IVs)))

        if forced_in is not None:
            if IV in forced_in:
                print("Skipping this variable, since it is being forced in already")
                continue

        if forced_in is not None:
            model_dataset = dataset[[DV, IV] + forced_in]
        else:
            model_dataset = dataset[[DV, IV]]

        model_dataset = model_dataset.dropna()
        kept_index = model_dataset.index.values

        if forced_in is not None:
            X = model_dataset[[IV] + forced_in]
        else:
            X = model_dataset[[IV]]
        Y = model_dataset[DV]

        results = model.fit(X, Y)

        # Generate outputs
        results_dict = OrderedDict()

        results_dict['Variable'] = IV
        if forced_in is not None:
            results_dict['Forced In'] = "|".join(forced_in)
        results_dict['# Obs'] = model_dataset.shape[0]
        results_dict['# Miss'] = len(Y) - model_dataset.shape[0]

        if has_intercept:
            results_dict['Intercept'] = results.intercept_[0]
        results_dict['Var Coef'] = results.coef_[0][results.feature_names_in_.tolist().index(IV)]

        if forced_in is not None:
            for forced_var in forced_in:
                results_dict[forced_var + " Coef"] = results.coef_[0][results.feature_names_in_.tolist().index(forced_var)]

        predicted = model.predict_proba(X)[:,1]

        probability_metrics, classification_metrics, confusion_matrix_df = generate_metrics(predicted_probs=predicted, labels=fitted_values['Actual'], threshold=threshold)
        
        for metric in ['roc_auc', 'average_precision', 'efron_r2', 'gini']:
            results_dict[metric] = probability_metrics[metric]
        
        for metric in ['accuracy', 'specificity', 'recall', 'precision', 'f1']:
            results_dict[metric] = classification_metrics[metric]
            
        results_dict['True Positive'] = confusion_matrix_df.iloc[1,1]
        results_dict['False Positive'] = confusion_matrix_df.iloc[1,0]
        results_dict['False Negative'] = confusion_matrix_df.iloc[0,1]
        results_dict['True Negative'] = confusion_matrix_df.iloc[0,0]

        output = pd.concat([output, pd.DataFrame(results_dict, index=[0])]).reset_index(drop=True)

        # Fitted values
        if get_fitted:
            fitted_values[IV] = np.nan
            fitted_values.loc[kept_index, IV] = predicted

    if get_fitted is False:
        fitted_values = None

    return output, fitted_values



def lasso_logistic_regression(dataset, DV, IVs, forced_in=None, intercept=True, C_list=None):
    '''
    Runs LASSO regression on data for variable selection purposes in logistic regression models and generates outputs for all LASSO
    Perform logistic regression on individual independent variables, with optional forced in variables
    :param dataset: pandas dataframe
    :param DV: name of target variable (dependent variable)
    :param IVs: list of names of independent variables
    :param forced_in: optional list of forced in variables. All variables named here will be forced in
    :param intercept: Boolean indicating whether or not to include intercept
    :param alpha_list: List of alpha penalty values
    :return: pandas dataset containing summary statistics and coefficients
    '''

    # Check inputs and reformat if necessary
    if DV is None:
        raise ValueError("You must include DV")
    if IVs is None:
        raise ValueError("You must include IVs")

    if forced_in is not None:
        if isinstance(forced_in, str):
            forced_in = [forced_in]
        IVs = list(set(IVs) - set(forced_in))

    if C_list is None:
        C_list = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 25]

    col_info = OrderedDict()
    col_info['C'] = pd.Series([], dtype='float')
    col_info['Variables'] = pd.Series([], dtype='str')
    col_info['Converged?'] = pd.Series([], dtype='bool')
    col_info['AUC'] = pd.Series([], dtype='float')

    if intercept:
        col_info['Intercept'] = pd.Series([], dtype='float')

    if forced_in is not None:
        vars = forced_in + IVs
    else:
        vars = IVs

    for var in vars:
        col_info["{} Coef".format(var)] = pd.Series([], dtype='float')

    output = pd.DataFrame(col_info)

    scaler = StandardScaler()

    if forced_in is not None:
        model_dataset = dataset[[DV] + IVs + forced_in].dropna()
        model_dataset[IVs + forced_in] = scaler.fit_transform(model_dataset[IVs + forced_in])
        X = model_dataset[forced_in + IVs].copy()
    else:
        model_dataset = dataset[[DV] + IVs].dropna()
        model_dataset[IVs] = scaler.fit_transform(model_dataset[IVs])
        X = model_dataset[IVs].copy()

    Y = model_dataset[DV]

    if forced_in is not None:
        X[forced_in] = X[forced_in] * 1000

    for C in C_list:
        logistic = LogisticRegression(penalty='l1', C=C, solver='liblinear', fit_intercept=intercept)

        fitted_model = logistic.fit(X, Y)

        results = OrderedDict()

        results['C'] = C
        results['Converged?'] = fitted_model.n_iter_[0] < fitted_model.max_iter  # Assuming max number of iterations met means no convergence

        Y_pred = fitted_model.predict_proba(X)[:, 1]
        results['AUC'] = roc_auc_score(Y, Y_pred)

        if intercept:
            results['Intercept'] = fitted_model.intercept_[0]

        var_list = []
        for var in vars:
            var_index = vars.index(var)
            var_coef = fitted_model.coef_[0][var_index]
            if var in forced_in:
                var_coef = var_coef * 1000
            if var_coef != 0:
                var_list.append(var)
            results["{} Coef".format(var)] = var_coef

        results['Variables'] = ";".join(var_list)

        output = pd.concat([output, pd.DataFrame(results, index=[0])]).reset_index(drop=True)

    return output

    

default_mfa_options = {} # Not currently used
def mfa_linear_model(model, dataset, DV, IVs, get_fitted=True, threshold=0.5, detailed=False, mfa_options=default_mfa_options):
    '''
    Perform linear model regression on a set of independent variables
    :param model: Instantiated model class from https://scikit-learn.org/stable/modules/linear_model.html
    :param dataset: pandas dataframe
    :param DV: name of target variable (dependent variable)
    :param IVs: list of names of independent variables to be evaluated one by one
    :param get_fitted: boolean whether to get fitted values
    :param detailed: boolean whether to produce detailed test results and charts. Currently set up to only produce VIF
    :param mfa_options: dictionary with test options. NOT CURRENTLY USED
    :return: dictionary with 'results' and 'fitted_values' (if requested)
    '''

    # Check if intercept
    has_intercept = model.fit_intercept

    # Check inputs and reformat if necessary
    if DV is None:
        raise ValueError("You must include DV")
    if IVs is None:
        raise ValueError("You must include IVs")

    if isinstance(IVs, str):
        IVs = [IVs]

    model_dataset = dataset[[DV] + IVs].dropna()
    kept_index = model_dataset.index.values

    X = model_dataset[IVs]
    Y = model_dataset[DV]

    results = model.fit(X,Y)
    predicted = model.predict_proba(X)[:, 1]
    
    # Generate outputs
    results_dict = OrderedDict()

    for i in range(len(IVs)):
        IV = IVs[i]
        results_dict['Var {}'.format(i + 1)] = IV
        results_dict['Var {} Coef'.format(i + 1)] = results.coef_[0][results.feature_names_in_.tolist().index(IV)]

    if has_intercept:
        results_dict['Intercept'] = results.intercept_

    results_dict['# Obs'] = model_dataset.shape[0]
    results_dict['# Miss'] = len(Y) - model_dataset.shape[0]
    
    probability_metrics, classification_metrics, confusion_matrix_df = generate_metrics(predicted_probs=predicted, labels=Y, threshold=threshold)

     # Incorporate metrics from generate_metrics function
    for metric in probability_metrics:
        results_dict[metric] = probability_metrics[metric]
    for metric in classification_metrics:
        results_dict[metric] = classification_metrics[metric]
    
    # Extract confusion matrix values
    results_dict["True Positives"] = confusion_matrix_df.iloc[1,1]
    results_dict["False Positives"] = confusion_matrix_df.iloc[1,0]
    results_dict["True Negatives"] = confusion_matrix_df.iloc[0,0]
    results_dict["False Negatives"] = confusion_matrix_df.iloc[0,1]

    # Statistical tests
    VIF = sklearn_vif(X)
    results_dict["Max_VIF"] = max(VIF['VIF'])

    if detailed:
        detailed_results = OrderedDict()
        detailed_results['VIF'] = VIF
        regression_object = model

    else:
        detailed_results = None
        regression_object = None

    # Fitted values
    if get_fitted:
        fitted_values = pd.DataFrame({"kept_index": kept_index, "fit": predicted})

    if get_fitted is False:
        fitted_values = None

    return {
        "summary": results_dict,
        "fitted": fitted_values,
        "detailed": detailed_results,
        "model": regression_object
    }
    
    
def mfa_linear_models(model, dataset, DV, IV_table, get_fitted=True, threshold=0.5, detailed=False, mfa_options=default_mfa_options):
    '''
    Perform OLS regression on individual independent variables, with optional forced in variables
    
    Perform linear model regression on a set of independent variables
    :param model: Instantiated model class from https://scikit-learn.org/stable/modules/linear_model.html
    :param dataset: pandas dataframe
    :param DV: name of target variable (dependent variable)
    :param IV_table: pandas table where each row contains the variables 
    :param get_fitted: boolean whether to get fitted values
    :param detailed: boolean whether to produce detailed test results and charts. Currently set up to only produce VIF
    :param mfa_options: dictionary with test options. NOT CURRENTLY USED
    in a model
    :return: dictionary with 'results' and 'fitted_values' (if requested)
    '''
    
    # Check if intercept
    has_intercept = model.fit_intercept

    # Check inputs and reformat if necessary
    if DV is None:
        raise ValueError("You must include DV")
    if IV_table is None:
        raise ValueError("You must include IV table")

    num_var = IV_table.shape[1]

    # Set up the result table in Pandas
    col_info = OrderedDict()

    # Variable names
    for i in range(num_var):
        col_info['Var {}'.format(i+1)] = pd.Series([], dtype='str')

    col_info['# Obs'] = pd.Series([], dtype='int')
    col_info['# Miss'] = pd.Series([], dtype='int')

    # Coefficients
    if has_intercept:
        col_info['Intercept'] = pd.Series([], dtype='float')
    for i in range(num_var):
        col_info['Var {} Coef'.format(i+1)] = pd.Series([], dtype='float')
    
    metric_names = ['accuracy', 'specificity', 'recall', 'precision', 'f1', 'roc_auc', 'average_precision', 'efron_r2', 'gini', 'True Positive', 'False Positive', 'False Negative', 'True Negative']
    
    for metric in metric_names:
        col_info['Var Coef'] = pd.Series([], dtype='float')
    
    # Tests
    col_info["Max_VIF"] = pd.Series([], dtype='float')

    # Create the pandas
    output = pd.DataFrame(col_info)

    # Create Pandas for fitted values
    fitted_values = dataset[[DV]].copy()
    fitted_values.columns = ["Actual"]

    # Loop through model table
    for i in range(IV_table.shape[0]):
    
        print("Working on model {} out of {}".format(i+1, IV_table.shape[0]))
        IV_list = IV_table.iloc[i,:]

        # Remove None and blanks
        IV_list = [x for x in IV_list if not pd.isnull(x)]
        IV_list = [x for x in IV_list if x != ""]
        
        try:
            # Execute main function
            reg_results = mfa_linear_model(model, dataset, DV, IV_list, get_fitted, threshold, detailed=False, mfa_options=mfa_options)
            
            # Add results to table
            output = pd.concat([output, pd.DataFrame(reg_results['summary'], index=[i])])
            
            # Fitted values
            if get_fitted:
                predictions = reg_results['fitted']
                fitted_values["Model {}".format(i+1)] = np.nan
                fitted_values.loc[predictions['kept_index'], "Model {}".format(i+1)] = predictions['fit']
        except:
            print("Model {} was not able to execute.".format(i+1))
            traceback.print_exc()

    if get_fitted is False:
        fitted_values = None

    return output, fitted_values
    



# Utilities -- for more statistical tests, use the statsmodels regression notebooks
def sklearn_vif(X_data):

    # Exogenous factors
    exogs = X_data.columns.tolist()
    
    # Initialize
    vifs = []

    if len(exogs) == 1:
        vifs = [1]
    
    else: 
        # form input data for each exogenous variable
        for exog in exogs:
            not_exog = [i for i in exogs if i != exog]
            X, y = X_data[not_exog], X_data[exog]

            # extract r-squared from the fit
            r_squared = linear_model.LinearRegression(fit_intercept=True).fit(X, y).score(X, y)

            # calculate VIF
            vif = 1/(1 - r_squared)
            vifs.append(vif)
        
    # return VIF DataFrame
    df_vif = pd.DataFrame({'Var': exogs, 'VIF': vifs})

    return df_vif



def get_LogisticRegressionCV_path(logistic_model, X, y, cv, threshold = 0.5):

    metrics_table = pd.DataFrame(logistic_model.Cs_, columns=['C'])

    # Initialize lists to store the metrics for each C
    accuracy_scores = []
    f1_scores = []
    auc_scores = []
    precision_scores = []
    non_zero_coefs = []
    for idx, C in enumerate(logistic_model.Cs_):
        print(f"Working on #{idx + 1} out of #{len(logistic_model.Cs_)} values of C")
        # Set the regularization parameter
        this_model = LogisticRegressionCV(Cs=[C], cv=cv, penalty='l1', solver='liblinear', random_state=1000, 
                                          fit_intercept=True, intercept_scaling=1000,
                                          refit=True).fit(X,y)

        # Calculate Accuracy and F1 score using predicted class labels
        y_proba = cross_val_predict(this_model, X, y, cv=cv, method='predict_proba')[:, 1]
        y_pred = np.where(y_proba > threshold, 1, 0)
        accuracy_scores.append(accuracy_score(y, y_pred))
        f1_scores.append(f1_score(y, y_pred, average='weighted'))

        # Calculate AUC and AUC-PR using probability estimates
        auc_scores.append(roc_auc_score(y, y_proba))
        precision_scores.append(average_precision_score(y, y_proba))
        
        non_zero_coefs.append(np.sum(this_model.coef_ != 0))
    
    metrics_table['Accuracy'] = accuracy_scores
    metrics_table['F1 Score'] = f1_scores
    metrics_table['Average precision'] = precision_scores
    metrics_table['AUC'] =  auc_scores
    metrics_table['# Variables'] =  non_zero_coefs
        
    return metrics_table


def get_ElasticNetCV_path(ENet_model, X, y, cv, threshold = 0.5):

    Cs = []
    l1_ratios = []

    accuracy_scores = []
    f1_scores = []
    auc_scores = []
    precision_scores = []
    non_zero_coefs = []

    for idx, l1_rat in enumerate(ENet_model.l1_ratios_):
        print(f"Working on #{idx + 1} out of #{len(ENet_model.l1_ratios_)} values of l1_ratios_")
        for jdx, C in enumerate(ENet_model.Cs_):
            # Set the regularization parameter
            this_model = LogisticRegressionCV(Cs=[C], l1_ratios=[l1_rat], cv=cv, penalty='elasticnet', solver='saga', 
                                              random_state=1000, 
                                              fit_intercept=True, intercept_scaling=1000,
                                              refit=True).fit(X,y)
            
            Cs.append(C)
            l1_ratios.append(l1_rat)

            # Calculate Accuracy and F1 score using predicted class labels
            y_proba = cross_val_predict(this_model, X, y, cv=cv, method='predict_proba')[:, 1]
            y_pred = np.where(y_proba > threshold, 1, 0)
            accuracy_scores.append(accuracy_score(y, y_pred))
            f1_scores.append(f1_score(y, y_pred, average='weighted'))

            # Calculate AUC and AUC-PR using probability estimates
            auc_scores.append(roc_auc_score(y, y_proba))
            precision_scores.append(average_precision_score(y, y_proba))
            
            non_zero_coefs.append(np.sum(this_model.coef_ != 0))
            
    metrics_table = pd.DataFrame({
        "C": Cs,
        "l1_ratio": l1_ratios,
        "accuracy": accuracy_scores,
        "f1": f1_scores,
        "auc": auc_scores,
        "precision": precision_scores,
        "non_zero_coefs": non_zero_coefs
    })

    return metrics_table

