import pickle
from datetime import datetime,timedelta
import urllib
import matplotlib.dates as mdates
import json 
import requests
import pandas as pd 
import numpy as np





def load_position(data_path,date_range,spc=["l1","solo"]):
    with open(data_path+'positions_from_2010_to_2030_HEEQ_10min_rad_ed.p', "rb") as f:
        pos = pickle.load(f)
    columns = pos['l1'].dtype.names
    positions = {}
    for s in spc:
        idx = np.argwhere(np.logical_and(pos[s][columns[0]]>=date_range[0],pos[s][columns[0]]<=date_range[1]))
        positions[s] = pos[s][idx]
     
    return positions


def load_donki(results_path,dates):

    url_donki='https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CMEAnalysis?startDate='+dates[0].strftime('%Y-%m-%d')+'&endDate='+dates[1].strftime('%Y-%m-%d')+'&mostAccurateOnly=true'
    try:
        r = requests.get(url_donki)
        with open(results_path+'DONKI.json','wb') as f:
            f.write(r.content)
    except:
        print(url_donki)
        print('DONKI not loaded')   

    f = open(results_path+'DONKI.json')
    data = json.load(f)
    CMEs = {}    
    for d in data:
        if(d["associatedCMEID"] in CMEs.keys()):
            d["time21_5"] = datetime.strptime(d["time21_5"],"%Y-%m-%dT%H:%MZ")
            for k in d.keys():
                if(isinstance(d[k], datetime)):
                    diff = timedelta(seconds=(CMEs[d["associatedCMEID"]][k]-d[k] ).total_seconds())
                    
                    if(CMEs[d["associatedCMEID"]][k]>d[k]):
                        CMEs[d["associatedCMEID"]][k] = CMEs[d["associatedCMEID"]][k] + diff/2
                    else:
                        CMEs[d["associatedCMEID"]][k] = d[k] + diff/2
                   
                elif(isinstance(d[k], (int, float, complex))):
                    CMEs[d["associatedCMEID"]][k] = np.nanmean([CMEs[d["associatedCMEID"]][k],d[k]])
        else:
            d["time21_5"] = datetime.strptime(d["time21_5"],"%Y-%m-%dT%H:%MZ")
            CMEs[d["associatedCMEID"]] = d


    return list(CMEs.values())


if __name__ == "__main__":
    print(len(load_donki("./")))