# Operations Layer
<em>This is the Operations Layer of the ADSDB Project.</em>

In this projet, you will use an interface (an executable script) to create the Data Management Backbone. This backbone is responsible for executing information from data sources and then storing it in databases. Simply, do all the steps from Landing Zone (temporal) to Exploitation Zone (like the jupyter notebooks).

The project is divided into the following sections:

### 1. Landing Zone
The Landing Zone is where the data sources are stored. This zone holds datasets (json format) extracted from various APIs sources. Since we have a larger number of games, extracting data from APIs could take a significant amount of time (up to several days). Therefore, a tempory folder is created in the landing zone to store a small part of the datasets for faster processing, as the full datasets contains over 150000 games.

### 2. Python_files:
The Python files folder contains python scripts, which are intially developed in jupyter notebooks. These scripts handled the creation of all necessary zones in the Data Management Backbone, which is responsible for extracting, transforming, loading and storing the data in the corresponding databases of each zone. Additionally, this folder includes: 
- **CheckCodeQuality.py**: this script is used to check the code quality of the python code files.
- **monitor_script.py**: a monitoring script to track the execution of Data Management Backbone in runtime.

### 3. Test_files:
The Test_files folder contains unit tests to validate the functionality of the python code files. For each script, there is a corresponding test script to ensure it performs as expected.

### 4. UI.py:
The UI.py is an executable interface with two primary functions and corresponding "terminals":
- **"Start Data Organization"** button: located on the left side of the interface, this button initiates the execution of the python code files to extract, transform, load and store the data in the databases. The left-side terminal displays the progress of the execution. Due to integration with translation and currency conversion APIs for data cleaning, the terminals may pause for a few seconds while waiting for the APIs to respond. Also, the right-side terminal displays the monitor of the execution in real time.
Once the execution is completed, in this project folder will create the zone folders where contain the databases with corresponding tables.


- **"Check Code Quality"** button: located on the right side of the interface, this button opens a new window with a "terminal" to show the code quality of all python code files.

### 5. Requirements.txt:
The requirements.txt file contains all the necessary libraries and packages required to run the python code files. 


## 

Install Instructions:
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

To run the unit tests, navigate to the Test_files folder and run the following command:
```
python -m unittest test_file_name.py
```