# Lagrangian-time-series-model-for-aerosol-prediction-and-driver-analysis
This repository contains MATLAB code for developing a Lagrangian time-series framework to predict aerosol concentrations and to interpret the trained model. The workflow includes air-mass trajectory calculation, input data extraction along trajectories, model development, and model interpretation.

The main scripts are organized as follows:

    1_Trajectory_calculation_and_read\
        traj_calulation: Calculates backwardairmass trajectories using the HYSPLIT model.
        traj_read: Reads trajectory output files and organizes them into MATLAB cell arrays for each year.
    2_Input_data_extraction\
        traj_data_organize: Organizes trajectory data and determines the land–ocean flag (flag_Land; 0 = ocean, 1 = land) for each trajectory point.
        MERRA_data_along_traj_10day: Extracts meteorological and aerosol-related variables from the MERRA-2 reanalysis along each airmass trajectory.
        DMS_data_along_traj_10day: Extracts sea-surface dimethyl sulfide (DMS) concentrations along each airmass trajectory.
        Chla_data_along_traj_10day: Extracts sea-surface chlorophyll-a concentrations along each airmass trajectory, accounting for airmass dispersion.
        SO2EM_data_along_traj_10day: Extracts SO2 emission flux data along each airmass trajectory, accounting for airmass dispersion.
        PREC_data_along_traj_10day: Extracts GPM precipitation rate data along each airmass trajectory, accounting for airmass dispersion.
    3_Model_development\
        Time-series_models\hyperparameter_grid_search_*_MCCV: Performs hyperparameter grid search for the time-series models, with and without implementing a weighted resampling strategy.
        Time-series_models\model_trainining_*_MCCV: Trains an ensemble of time-series models using the optimal hyperparameter combination.
        Time-series_models\LSTM_10_model_ensemble.mat: The trained 10-LSTM model ensemble with implementing the weighted resampling.
        Time-series_models\LSTM_predict: Predict the aerosol concentration using the developed LSTM ensemble model.
        Traditional_models\*_grid_search: Conducts hyperparameter grid search for each non-sequential machine learning model.
        Traditional_models\*_training: Trains each non-sequential machine learning model using the selected optimal hyperparameters.
    4_Model_interpretation\
        GSA_Sobol_indice: Calculates the feature importance (Sobol' indice) of each input variable to both total target variability and intramonthly variability.
        GSA_for_monthly_variation: Calculates the feature importance (Sobol' indice) of each input variable for the intermonthly variability of the target variable.
        GSA_time_series: Investigates the temporal evolution of feature importance along airmass trajectories for each input variable.

To run these scripts, MATLAB Statistics and Machine Learning Toolbox and Deep Learning Toolbox are required. Python is needed for training XGBoost and LightGBM models.

Please feel free to contact Shengqian Zhou (shengqian@wustl.edu) for any questions regarding the code.
