import numpy as np 
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import astropy.units as u
import data_utils
from sunpy.time import parse_time
import multiprocessing
import time
import code_base.wrappers as wrappers


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2)           
    theta = np.arctan2(z,np.sqrt(x**2+ y**2))
    phi = np.arctan2(y,x)                    
    return r, theta, phi


def process_arrival(distance, obj, time1, cme_v, cme_id, t0, halfAngle, speed, cme_lon, cme_lat, label):
        arr_time = []
        arrival = []
        arr_time_fin = []
        arr_time_err0 = []
        arr_time_err1 = []
        arr_id = []
        arr_hit = []
        arr_speed_list = []
        arr_speed_err_list = []

        if not np.isnan(distance).all():
            if label == 'earth':
                for t in range(3):
                    index = np.argmin(np.abs(np.ma.array(distance[:, t], mask=np.isnan(distance[:, t])) - obj))
                    arr_time.append(time1[int(index)])

            else:
                for t in range(3):
                    index = np.argmin(np.abs(distance[:,t] - obj))
                    arr_time.append(time1[int(index)])

            arr_speed = cme_v[:, 0][index]
            err_arr_speed = cme_v[:, 2][index] - cme_v[:, 1][index]
            err_arr_time = (arr_time[1] - arr_time[2]).total_seconds() / 3600.0
            arrival.append([
                cme_id[0].decode("utf-8"),
                t0.strftime('%Y-%m-%dT%H:%MZ'),
                "{:.1f}".format(cme_lon[0]),
                "{:.1f}".format(cme_lat[0]),
                "{:.1f}".format(halfAngle),
                "{:.1f}".format(speed),
                arr_time[0].strftime('%Y-%m-%dT%H:%MZ'),
                "{:.2f}".format(err_arr_time / 2),
                "{:.2f}".format(arr_speed),
                "{:.2f}".format(err_arr_speed / 2)
            ])
            arr_time_fin.append(arr_time[0])
            arr_time_err0.append(arr_time[0] - timedelta(hours=err_arr_time))
            arr_time_err1.append(arr_time[0] + timedelta(hours=err_arr_time))
            arr_id.append(cme_id[0].decode("utf-8"))
            arr_hit.append(1.0)
            arr_speed_list.append(arr_speed)
            arr_speed_err_list.append(err_arr_speed / 2)
        
        else:
            arr_time_fin.append(np.nan)
            arr_time_err0.append(np.nan)
            arr_time_err1.append(np.nan)
            arr_id.append(np.nan)
            arr_hit.append(np.nan)
            arr_speed_list.append(np.nan)
            arr_speed_err_list.append(np.nan)


        return {
            f"arrival": arrival,
            f"arr_time_fin": arr_time_fin,
            f"arr_time_err0": arr_time_err0,
            f"arr_time_err1": arr_time_err1,
            f"arr_id": arr_id,
            f"arr_hit": arr_hit,
            f"arr_speed_list": arr_speed_list,
            f"arr_speed_err_list": arr_speed_err_list,
        }

def Prediction_ELEvo(time21_5, latitude, longitude, halfAngle, speed, type, isMostAccurate, associatedCMEID, associatedCMEstartTime, note, associatedCMELink, catalog, featureCode, dataLevel, measurementTechnique, imageType, tilt, minorHalfWidth, speedMeasuredAtHeight, submissionTime, versionId, link,positions):
    print(associatedCMEID)
    distance0 = 21.5*u.solRad.to(u.km)
    t0 = time21_5
    gamma_init = 0.1
    ambient_wind_init = 400.
    kindays = 15
    n_ensemble = 50000
    halfwidth = np.deg2rad(halfAngle)
    res_in_min = 10
    f = 0.7
    kindays_in_min = int(kindays*24*60/res_in_min)

    earth = positions["l1"]

    

    ### Just doing earth for now, if needed create generic function and call it with spacecraft pos array

    # find index at which list of earth times corresponds to time at 21.5 Rs of current CME
    dct = mdates.date2num(time21_5) - earth.time
    earth_ind = np.argmin(np.abs(dct))
    

    # convert longitude to radians and calculate delta_earth (probably not necessary since longitude of Earth is close to 0 all the time)
    if np.abs(np.deg2rad(longitude)) + np.abs(earth.lon[earth_ind][0]) > np.pi and np.sign(np.deg2rad(longitude)) != np.sign(earth.lon[earth_ind][0]):
        delta_earth = np.deg2rad(longitude) - (earth.lon[earth_ind][0] + 2 * np.pi * np.sign(np.deg2rad(longitude)))
    else:
        delta_earth = np.deg2rad(longitude) - earth.lon[earth_ind][0]
   

    # times for each event kinematic
    time1=[]
    tstart1=time21_5
    tend1=tstart1+timedelta(days=kindays)

    # make 30 min datetimes
    while tstart1 < tend1:

        time1.append(tstart1)  
        tstart1 += timedelta(minutes=res_in_min)    

   
    # initialize arrays for kinematics
    timestep=np.zeros([kindays_in_min,n_ensemble])
    cme_r=np.zeros([kindays_in_min, 3])
    cme_v=np.zeros([kindays_in_min, 3])
    cme_lon=np.ones(kindays_in_min)*longitude
    cme_lat=np.ones(kindays_in_min)*latitude
    cme_id=np.chararray(kindays_in_min, itemsize=27)
    cme_id[:]=associatedCMEID
    cme_r_ensemble=np.zeros([kindays_in_min, n_ensemble])
    cme_v_ensemble=np.zeros([kindays_in_min, n_ensemble])
    

    cme_delta=delta_earth*np.ones([kindays_in_min,3])

    cme_hit=np.zeros(kindays_in_min)
    cme_hit[np.abs(delta_earth)<halfwidth] = 1


    distance_earth = np.empty([kindays_in_min,3])
    distance_solo = np.empty([kindays_in_min,3])
    distance_sta = np.empty([kindays_in_min,3])
    distance_earth[:] = np.nan
    distance_solo[:] = np.nan
    distance_sta[:] = np.nan


        
    kindays_in_min = int(kindays*24*60/res_in_min)
    
    gamma = np.abs(np.random.normal(gamma_init,0.025,n_ensemble))
    ambient_wind = np.random.normal(ambient_wind_init,50,n_ensemble)
    speed_ensemble = np.random.normal(speed,50,n_ensemble)
    
    timesteps = np.arange(kindays_in_min)*res_in_min*60
    timesteps = np.vstack([timesteps]*n_ensemble)
    timesteps = np.transpose(timesteps)

    accsign = np.ones(n_ensemble)
    accsign[speed_ensemble < ambient_wind] = -1.

    distance0_list = np.ones(n_ensemble)*distance0
    
    cme_r_ensemble = (accsign / (gamma * 1e-7)) * np.log(1 + (accsign * (gamma * 1e-7) * ((speed_ensemble - ambient_wind) * timesteps))) + ambient_wind * timesteps + distance0_list
    cme_v_ensemble = (speed_ensemble - ambient_wind) / (1 + (accsign * (gamma * 1e-7) * (speed_ensemble - ambient_wind) * timesteps)) + ambient_wind

    cme_r_mean = cme_r_ensemble.mean(1)
    cme_r_std = cme_r_ensemble.std(1)
    cme_v_mean = cme_v_ensemble.mean(1)
    cme_v_std = cme_v_ensemble.std(1)
    cme_r[:,0]= cme_r_mean*u.km.to(u.au)
    cme_r[:,1]=(cme_r_mean - 2*cme_r_std)*u.km.to(u.au) 
    cme_r[:,2]=(cme_r_mean + 2*cme_r_std)*u.km.to(u.au)
    cme_v[:,0]= cme_v_mean
    cme_v[:,1]=(cme_v_mean - 2*cme_v_std)
    cme_v[:,2]=(cme_v_mean + 2*cme_v_std)
    
    #Ellipse parameters   
    theta = np.arctan(f**2*np.ones([kindays_in_min,3]) * np.tan(halfwidth*np.ones([kindays_in_min,3])))
    omega = np.sqrt(np.cos(theta)**2 * (f**2*np.ones([kindays_in_min,3]) - 1) + 1)   
    cme_b = cme_r * omega * np.sin(halfwidth*np.ones([kindays_in_min,3])) / (np.cos(halfwidth*np.ones([kindays_in_min,3]) - theta) + omega * np.sin(halfwidth*np.ones([kindays_in_min,3])))    
    cme_a = cme_b / f*np.ones([kindays_in_min,3])
    cme_c = cme_r - cme_b
        
    root = np.sin(cme_delta)**2 * f**2*np.ones([kindays_in_min,3]) * (cme_b**2 - cme_c**2) + np.cos(cme_delta)**2 * cme_b**2
    distance_earth[cme_hit.all() == 1] = (cme_c * np.cos(cme_delta) + np.sqrt(root)) / (np.sin(cme_delta)**2 * f**2*np.ones([kindays_in_min,3]) + np.cos(cme_delta)**2) #distance from SUN in AU for given point on ellipse


    #### linear interpolation to 10 min resolution

    #find next full hour after t0
    format_str = '%Y-%m-%d %H'  
    t0r = datetime.strptime(datetime.strftime(t0, format_str), format_str) +timedelta(hours=1)
    time2=[]
    tstart2=t0r
    tend2=tstart2+timedelta(days=kindays)
    #make 30 min datetimes 
    while tstart2 < tend2:
        time2.append(tstart2)  
        tstart2 += timedelta(minutes=res_in_min)  

    time2_num=parse_time(time2).plot_date        
    time1_num=parse_time(time1).plot_date
    

  
    #linear interpolation to time_mat times    
    cme_r =  np.array([np.interp(time2_num, time1_num,cme_r[:,i]) for i in range(3)]).transpose()
    cme_v = np.array([np.interp(time2_num, time1_num,cme_v[:,i]) for i in range(3)]).transpose()
    cme_lat = np.array(np.interp(time2_num, time1_num,cme_lat )).transpose()
    cme_lon = np.array(np.interp(time2_num, time1_num,cme_lon )).transpose()
    cme_a = np.array([np.interp(time2_num, time1_num,cme_a[:,i]) for i in range(3)]).transpose()
    cme_b = np.array([np.interp(time2_num, time1_num,cme_b[:,i]) for i in range(3)]).transpose()
    cme_c = np.array([np.interp(time2_num, time1_num,cme_c[:,i]) for i in range(3)]).transpose()
    distance_earth = np.array(distance_earth).transpose()

    results_earth = process_arrival(distance_earth, earth.r[earth_ind][0], time1, cme_v, cme_id, t0, halfAngle, speed, cme_lon, cme_lat, label="earth")

    return time2_num, cme_r, cme_lat, cme_lon, cme_a, cme_b, cme_c, cme_id, cme_v#, results_earth

def fun_wrapper(dict_args):
    return Prediction_ELEvo(**dict_args)

if __name__ == '__main__':

    path = "/Users/maikebauer/Code/ELEvoHI_Python/code/"
    liste_spc = ['l1','solo','psp','sta','bepi','mercury','venus','mars']
    dates = [datetime(2024,5,1),datetime(2024,5,20)]
    data = data_utils.load_donki(path,dates)
    positions = data_utils.load_position(path,[mdates.date2num(dates[0]),mdates.date2num(dates[1])],liste_spc)
    
    for d in data:
       d["positions"]=positions
   
   


    print('Generating kinematics using ELEvo')

    start_time = time.time()

    if len(data) >= 5:
        used=5
    else:
        used=1
    
    pool = multiprocessing.Pool(used)
    results = pool.map(fun_wrapper, data)
    pool.close()
    pool.join()