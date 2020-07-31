# import modules
import json
import os

def GetDict(file_name):
    
    """
    params: filename
        filename of config file
        
    returns: config_dict (json serialized dictionary for config content)
    
    """
    with open(os.path.join('../config',file_name)) as config:
        config_dict = json.load(config)
        
    return config_dict

def SetDict(file_name, json_data):
    
    """
    params: 
        filename
            filename of config file
        json_data
            data from confit file
        
    returns: None
    
    """
                                                               
    with open(os.path.join('../config',file_name), "w") as file:
        file.write(json_data)                           
        
