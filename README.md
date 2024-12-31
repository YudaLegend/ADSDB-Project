# Operations Layer
<em>This is the Operations Layer of the ADSDB Project.</em>

In this projet, you will use an interface (an executable script) to create the `Data Management Backbone` and the `Data Analysis Backbone`. These backbones are responsible for managing data from its extraction to analysis and modeling.

- **Data Management Backbone:** Handles the movement of data from the Landing Zone (temporary storage) to the Exploitation Zone (analysis-ready datasets).

- **Data Analysis Backbone:** Prepares data for feature engineering and modeling, evaluates model performance, and facilitates hyperparameter tuning and feature selection.

## Project Structure

### 1. Data Management

The `Data Management` process involves organizing and processing data across multiple zones. The primary folder in this directory is the `Landing Zone`, which serves as the storage area for raw data extracted from various APIs in JSON format. Given the large volume of data, over 150,000 games. API extraction can take significant time (potentially several days). To streamline the process, a temporary folder is created within the `Landing Zone` to store a subset of the datasets for quicker testing and processing.

The remaining zones of the `Data Management Backbone` are dynamically created during the execution of the corresponding Python scripts, which handle the extraction, transformation, loading, and organization of the data into appropriate formats for subsequent analysis.

### 2. Python_files:

This folder contains scripts for both the `Data Management` and `Data Analysis` processes, as well as additional utility scripts.

The scripts manage the creation of all necessary zones in the `Data Management Backbone`, which supervises the extraction, transformation, loading, and storage of data into the corresponding databases for each zone. Similarly, the scripts facilitate the creation of all zones in the `Data Analysis Backbone`, responsible for extracting, transforming, preparing, and splitting features required for model creation. They also handle model evaluation using appropriate techniques to ensure optimal performance.

The scripts are divided into the following folders:

- **Check_Code:** the folder contains the `CheckCodeQuality.py`, that is used to check the code quality of the python code files.

- **Data_Analysis:** the folder contains all python files related to the Data Analysis Backbone. It contains the following folders:

    - **AnalyticalSandbox_Zone:** the folder contains the scripts that corresponding to the `AnalyticalSandbox Zone` of the `Data Analysis Backbone`, which is responsible to extracting the useful features from the database of `Exploitation Zone` and then creating a new table with the selected features in `analytical_sandbox.duckdb`.

    - **FeatureEngineering_Zone:** the folder contains the scripts that corresponding to the `FeatureEngineering Zone` of the `Data Analysis Backbone`, which is responsible to transforming, creating, imputing and preparing the features of the `analytical_sandbox.duckdb` database, then creating a new table with cleaned features and creating a train/test split set of features in `feature_engineering.duckdb`. 

    - **ModelGeneration_Zone:** the folder contains the scripts that corresponding to the `ModelGeneration Zone` of the `Data Analysis Backbone`, which is responsible to building various models using the prepared table in `feature_engineering.duckdb` database. Once all models are created and saved, need be to evaluated using validation metrics. Also, including a feature selection to keep the significant features and a hyperparameter tuning to find the optimal parameters for the best model for the given dataset.

- **Data_Management:** the folder contains all python files related to the `Data Management Backbone`. It contains the following folders:

    - **Exploitation_Zone:** the folder contains the scripts that corresponding to the `Exploitation Zone` of the `Data Management Backbone`, which is responsible to merging the data from different sources and creating KPIs for the data in the `exploitation_zone.duckdb`.

    - **Formatted_Zone:** the folder contains the scripts that corresponding to the `Formatted Zone` of the `Data Management Backbone`, which is responsible to format the data from the various sources stored in `Data Management/Landing Zone/Persistent` and create tables for the corresponding datasources in the `formatted_zone.duckdb`.

    - **Landing_Zone:** the folder contains the scripts that corresponding to the `Data Management/Landing Zone` of the `Data Management Backbone`, which is responsible to extract the data from various sources using API and store it in the landing zone, specifically store the raw data in `Data Management/Landing Zone/Temporal`, and then store persistently these data in `Data Management/Landing Zone/Persistent` organized by folders corresponding to each source.

    - **Trusted_Zone:** the folder contains the scripts that corresponding to the `Trusted Zone` of the `Data Management Backbone`, which is responsible to merge the different versions of each datasource into one, and make a data quality to remove any duplicates or inconsistencies data. Also, the neccesary transformations are performed to the data and the final data is stored in the `trusted_zone.duckdb`.

- **Monitoring:** the folder contains the **monitor_script.py**, that is used to monitor the execution of the `Data Management Backbone` and `Data Analysis Backbone` in runtime.


### 3. Test_files:
The `Test_files` folder contains unit tests to validate the functionality of the python code files. So, the scripts are divided into the following folders:

- **Data Analysis Test:** the folder contains all test scripts for the `Data Analysis Backbone`. For each script, there is a corresponding test script to ensure it performs as expected.

    - **AnalyticalSandbox_Test:** the folder contains the corresponding test scripts for all the python scripts in the `AnalyticalSandbox_Zone` folder.

    - **FeatureEngineering_Test:** the folder contains the corresponding test scripts for all the python scripts in the `FeatureEngineering_Zone` folder.

    - **ModelGeneration_Test:** the folder contains the corresponding test scripts for all the python scripts in the `ModelGeneration_Zone` folder.

- **Data Management Test:** the folder contains all test scripts for the `Data Management Backbone`. For each script, there is a corresponding test script to ensure it performs as expected. 

    - **Exploitation_Test:** the folder contains the corresponding test scripts for all the python scripts in the `Exploitation_Zone` folder.

    - **Formatted_Test:** the folder contains the corresponding test scripts for all the python scripts in the `Formatted_Zone` folder.

    - **Landing_Test:** the folder contains the corresponding test scripts for all the python scripts in the `Landing_Zone` folder.

    - **Trusted_Test:** the folder contains the corresponding test scripts for all the python scripts in the `Trusted_Zone` folder.


### 4. UI.py:
The UI.py is an executable interface with two primary functions and corresponding "terminals":

- **"Start Data Management" Button**

    - **Activation:** Activate by default.

    - **Function:** Executes Python scripts to perform the following tasks:
        - Create folders for `Data Management` process.

        - Extract, transform, convert, clean the data for posterior analysis.

        - Store the data cleaned in the appropriate database of each zone folder.

    - **Display:** 
        - The left side terminal displays the progress of the data management process.

        - The right side terminal displays the monitor of the execution process in real time.
    
    - **Additional Notes:** Integration with translation and currency conversion APIs for data cleaning may cause occasional pauses as the interface waits for API responses.

    - **Outcome:** Once the execution process is completed. The `Data Management` folder contains the zone folders, which each of them contains the corresponding database and tables. Then the `Start Data Analysis` button becomes enabled.


- **"Start Data Analysis" Button**

    - **Activation:** Enabled after the successful execution of the `Data Management` process.

    - **Function:** Executes Python scripts to perform the following tasks:
        - Create folders for `Data Analysis` process.

        - Extract, transform, prepare, and split features for model creation.

        - Store processed data in the corresponding databases.

        - Train and evaluate models using appropriate validation techniques.

    - **Display:**
        - The left side terminal displays the progress of the data analysis process.

        - The right side terminal displays the monitor of the execution process in real time.

        - Show the plots in a new window for better visualization and interpretation.

    - **Additional Notes:** Hyperparameter tuning, required to identify the best model parameters, may result in longer execution times due to the large number of parameter combinations (5 minutes approximately). Also, closing the plot windows to continue the execution.

    - **Outcome:** Once the execution process is finished. The zone folders are created in the `Data Analysis` folder containing the respective databases and tables. Moreover, trained models are saved in the `Data Analysis/ModelGeneration Zone/Models` folder including models from single train/test set, cross validation sets. Also the best model after performing all the process is saved. Then the `Start Prediction` button becomes enabled.


- **"Start Prediction" Button**
    
    - **Activation:** Enabled after the successful completion of both the `Data Management` and `Data Analysis` processes.

    - **Function:**
        - Creates a `Test Set` folder in the `Data Analysis` directory to store all test sets in CSV format.

        - Create a `Best Model Predict Set` folder in the `Data Analysis` directory to store predictions from different test sets.

        - Open a window to allow the selection of a CSV file to load. The window starts in the same directory as this `README.md` file. Navigate to the `Data Analysis/Test Set` folder to select the CSV file.

        - Uses the selected file to make predictions and validate against the truth set.

        - Stores the prediction results in the `Data Analysis/Best Model Predict Set` folder.

    - **Display:**
        - The left side terminal displays the different metric results of the prediction.

    - **Additional Notes:** Predictions can be repeated multiple times, where the predictions on the same test set will replace the existing results in the `Data Analysis/Best Model Predict Set` folder.

    - **Outcome:** The predict sets are returned and created in the `Data Analysis/Best Model Predict Set` folder, which corresponding to its test sets. Also, all test sets are saved in the `Data Analysis/Test Set` folder.


- **"Check Code Quality" Button**

    - **Activation:** Enabled by default.

    - **Function:** Show the code quality of all python code files, including `UI.py` and scripts in `Python_files`.

    - **Display:** Open a new "terminal" window to display the code quality of all python code files.

### 5. Requirements.txt:
The requirements.txt file contains all the necessary libraries and packages required to run the python code files. 


## Install Instructions
To run the project, you need to have Python installed on your machine.

To install Python (version >= 3.10.0), follow the instructions on the official website: https://www.python.org/downloads/

After installing Python, you need to install the necessary libraries and packages required to run the project using the following command:
```
pip3 install -r requirements.txt
```

To execute the interface, simply run the UI.py file using the following command in the terminal:
```
python UI.py
```

To run the unit tests, navigate to the Test_files/Zone_Test folder and run the corresponding test script using the following command:
```
python -m unittest test_file_name.py
```
