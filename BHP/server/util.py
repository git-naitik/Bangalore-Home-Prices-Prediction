import json
import pickle
import numpy as np
import os

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    try:  
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
        
    return round(__model.predict([x])[0])

def get_location_names():
    return __locations

def load_saved_artifacts(artifacts_path):
    print("Loading saved artifacts...start")
    global __data_columns
    global __locations
    
    with open(os.path.join(artifacts_path, "columns.json"), 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]
    
    global __model    
    with open(os.path.join(artifacts_path, "bangalore_home_prices_model"), 'rb') as f:
        __model = pickle.load(f)

if __name__ == "__main__":
    # Get the absolute path to the directory where util.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load saved artifacts using the absolute path
    load_saved_artifacts(os.path.join(base_dir, "artifacts"))
    
    print(get_location_names())
