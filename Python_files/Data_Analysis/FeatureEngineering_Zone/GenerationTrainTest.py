# This file is responsible for generating the train and test sets from the feature_steam_games table.
import duckdb
from sklearn.model_selection import train_test_split

# Generate train and test sets from the feature_steam_games table
def generationTrainTest(con):
    try:
        df = con.execute('SELECT * FROM feature_steam_games').df()
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        con.execute('CREATE TABLE IF NOT EXISTS train_dataset AS SELECT * FROM train')
        con.execute('CREATE TABLE IF NOT EXISTS test_dataset AS SELECT * FROM test')
    except Exception as e:
        print("An error occurred while generating train and test sets: ", e)
        return False
    return True


