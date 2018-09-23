# Disaster Response Pipeline Project

#### A pipeline that analyzes and categorizes messages sent during disaster events.  The project consists of an ETL and ML pipeline.  

#### The ETL pipeline can be found in the `data/process_data.py` file and it handles the following tasks:

* Combines the two given datasets
* Cleans the data
* Stores it in a SQLite database

#### The ML pipeline can be found in the `models/train_classifier.py` file and it handles the following tasks:

* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports your final model as a pickle file

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


#### Other files

* df.db - SQLite database
* model - Pickled classification model
* settings.py - Project Settings