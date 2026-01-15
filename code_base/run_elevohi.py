from code_base import functions
import numpy as np
import pandas as pd
from natsort import natsorted
import glob
from scipy import interpolate,optimize
import datetime
import os
from sunpy.coordinates import frames, get_horizons_coord
from sunpy.coordinates import Helioprojective, HeliocentricEarthEcliptic, HeliographicStonyhurst
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import constants as const
import matplotlib.pyplot as plt

def get_target_prop(name,thi):
    coord = get_horizons_coord(name, thi)
    # sc_hee = coord.transform_to(frames.HeliocentricEarthEcliptic)  #HEE
    sc_heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
    time = sc_heeq.obstime.to_datetime()
    r = sc_heeq.radius.value
    lon = np.deg2rad(sc_heeq.lon.value)
    lat = np.deg2rad(sc_heeq.lat.value)
    return time,r,lon,lat

def get_obj_heeq_pos(pos_time, obj_name):
    """
    Get the object's positions transformed to the Heliographic Stonyhurst frame.

    Parameters
    ----------
    pos_time : array_like
        Sequence of times at which to evaluate the object's position. Typically an
        array or list of astropy.time.Time (or a time-like object accepted by
        get_horizons_coord). The same object is returned unchanged.
    obj_name : str
        Name/identifier of the solar-system object as accepted by
        get_horizons_coord.

    Returns
    -------
    tuple
        A 4-tuple (pos_time, r, lon, lat) where:
        - pos_time : same type as the input pos_time (returned unchanged)
        - r : numpy.ndarray
            Radial distances corresponding to each time in pos_time. The values are
            returned as floats.
        - lon : numpy.ndarray
            Heliographic longitudes (in degrees) for each time, returned as floats.
        - lat : numpy.ndarray
            Heliographic latitudes (in degrees) for each time, returned as floats.

    Notes
    -----
    - Longitude and latitude are explicitly converted to degrees before being
      returned. Radial distances are returned as numeric values without units.
    """
    # get object position at given time
    try:
        iter(pos_time)
    except TypeError:
        pos_time = [pos_time]

    obj_pos = get_horizons_coord(obj_name, pos_time)

    # transform object position to Heliocentric Earth Equatorial coordinates
    obj_heeq_lon = np.array([SkyCoord(pos).transform_to(HeliographicStonyhurst()).lon.to(u.rad).value for pos in obj_pos])
    obj_heeq_lat = np.array([SkyCoord(pos).transform_to(HeliographicStonyhurst()).lat.to(u.rad).value for pos in obj_pos])
    obj_heeq_r = np.array([SkyCoord(pos).transform_to(HeliographicStonyhurst()).radius.to(u.AU).value for pos in obj_pos])

    return pos_time, obj_heeq_r, obj_heeq_lon, obj_heeq_lat

def load_strudl_tracks(path, return_parameters=False):
    
    times  = []
    elongs = []

    if return_parameters:
        params = []

    converted_strudl_dict = np.load(path, allow_pickle=True).item()

    for cme_key in converted_strudl_dict.keys():
        
        times.append([pd.to_datetime(d) for d in converted_strudl_dict[cme_key]["time"]])
        elongs.append(list(converted_strudl_dict[cme_key]["elongation"]))

        if return_parameters:
            phi = converted_strudl_dict[cme_key]["phi"]
            halfwidth = converted_strudl_dict[cme_key]["halfwidth"]
            vinit = converted_strudl_dict[cme_key]["vinit"]
            L1_ist_obs = converted_strudl_dict[cme_key]["L1_ist_obs"]
            cmeID_elevo = converted_strudl_dict[cme_key]["cmeID_elevo"]

            params.append({'phi':phi, 'halfwidth':halfwidth, 'vinit':vinit, 'L1_ist_obs':L1_ist_obs, 'cmeID_strudl':cme_key, 'cmeID_elevo':cmeID_elevo})

    if return_parameters:
        return times,elongs,params
    else:
        return times,elongs


def combine_tracks(track_times,track_elongs):

    min_times    = min(np.concatenate(track_times,0))
    max_times    = max(np.concatenate(track_times,0))
    new_time_axis = functions.calculate_new_time_axis(min_times, max_times, cadence=40)

    interpolated_elongs = []
    for t in range(0,len(track_times)):
        times = track_times[t]
        x     = [(times[i] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s') for i in range(0,len(times))]
        times = [(times[i]-min_times).total_seconds() for i in range(0,len(times))]
        interpf = interpolate.interp1d(x, track_elongs[t],kind='linear', fill_value="extrapolate")
        x_new = (new_time_axis - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        interpolated_elongs.append(interpf(x_new))
    

    return new_time_axis, np.array(interpolated_elongs).mean(0)

def interpolate_tracks(track_times,track_elongs,starttime=None,endtime=None):

    if starttime is None:
        min_time = min(track_times)
    else:
        min_time = starttime

    if endtime is None:
        max_time = max(track_times)
    else:
        max_time = endtime

    new_time_axis = functions.calculate_new_time_axis(min_time, max_time, cadence=40)

    x     = [(track_times[i] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s') for i in range(0,len(track_times))]

    interpf = interpolate.interp1d(x, track_elongs, kind='linear', fill_value="extrapolate")
    x_new = (new_time_axis - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    interpolated_elongs = interpf(x_new)

    return new_time_axis, np.array(interpolated_elongs)


def compute_delta(target_lon, direction):
    if abs(direction) + abs(target_lon) < np.pi:
        delta = direction - target_lon
    else:
        delta = direction - (target_lon + 2 * np.pi * np.sign(direction))

    return delta

def does_cme_hit_elevohi(delta, halfwidth):

    if round(np.rad2deg(np.abs(delta)), 2) < round(np.rad2deg(halfwidth), 2):
        hit = 1
    else:
        hit = 0

    return hit

def does_cme_hit(delta, halfwidth, tol=0.0):
    tol_rad = np.deg2rad(tol)

    return np.abs(delta) < (halfwidth + tol_rad)

def get_constants():
    AU_in_km = const.au.to_value('km')
    rsun_in_km = const.R_sun.to_value('km')

    return AU_in_km, rsun_in_km

def run_elevohi_baseline(track_times, track_elongs, track_parameters):

    AU_in_km, rsun_in_km = get_constants()

    f = track_parameters['f'][0]
    halfwidth = track_parameters['halfwidth'][0]

    basic_path = track_parameters['basic_path']
    pred_path = basic_path + 'ELEvoHI_Python/predictions/'
    eventdate = track_parameters['eventdate']
    HIobs = track_parameters['HIobs']
    prediction_path = pred_path + eventdate + '_' + HIobs + '/'

    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)

    endtime = datetime.datetime.strptime(track_parameters['endtime'], "%Y-%m-%d %H:%M")
    starttime = datetime.datetime.strptime(track_parameters['starttime'], "%Y-%m-%d %H:%M")

    timeaxis, tracks_interpolated = interpolate_tracks(track_times, track_elongs, starttime=starttime, endtime=endtime)

    startcut = timeaxis.searchsorted(starttime)
    endcut = timeaxis.searchsorted(endtime) - 1

    # ELEvoHI ensemble
    # Define the range of values for each parameter to build the ensemble
    if track_parameters['do_ensemble']:
        hw_range = [track_parameters['halfwidth'][1], track_parameters['halfwidth'][2]]
        hw_step = np.deg2rad(track_parameters['halfwidth_step'])
        #p_range = [track_parameters['phi_manual'][1], track_parameters['phi_manual'][2]]
        p_step = np.deg2rad(track_parameters['phi_step'])
        f_range = [track_parameters['f'][1], track_parameters['f'][2]]
        f_step = track_parameters['f_step']

    if track_parameters['phi_FPF']:
        raise NotImplementedError("FPF-based phi determination not implemented in this function.")
    
        # fpf_fit = fpf(track, startcut, endcut, prediction_path)
        # fpf_fit = fpf_mab(track, startcut, endcut, prediction_path)
        # phi = fpf_fit['phi_FPF']
        # if do_ensemble:
        #     start_phi = np.deg2rad(fpf_fit['phi_FPF'] - track_parameters['phi_FPF_range'][0])
        #     end_phi = np.deg2rad(fpf_fit['phi_FPF'] + track_parameters['phi_FPF_range'][1])
        #     num_points_phi = int(round((np.rad2deg(end_phi) - np.rad2deg(start_phi))/np.rad2deg(p_step) + 1))
        
    else:
        phi = track_parameters['phi_manual'][0]
        if track_parameters['do_ensemble']:
            start_phi = np.deg2rad(track_parameters['phi_manual'][1])    
            end_phi = np.deg2rad(track_parameters['phi_manual'][2])
            num_points_phi = int(round((track_parameters['phi_manual'][2] - track_parameters['phi_manual'][1])/np.rad2deg(p_step) + 1))

    phi = np.deg2rad(phi)
    halfwidth = np.deg2rad(halfwidth)
    f = track_parameters['f'][0]

    det_run = [phi, f, halfwidth]

    if track_parameters['do_ensemble']:
        start_lambda = np.deg2rad(hw_range[0])
        end_lambda = np.deg2rad(hw_range[1])
        num_points_lambda = int((hw_range[1] - hw_range[0])/np.rad2deg(hw_step) + 1)
        
        start_f = f_range[0]
        end_f = f_range[1]
        num_points_f = int((end_f - start_f)/f_step + 1)
        
        lambda_range = np.linspace(start_lambda, end_lambda, num_points_lambda)
        phi_range = np.linspace(start_phi, end_phi, num_points_phi)
        aspect_range = np.linspace(start_f, end_f, num_points_f)
                
        # Create a grid of parameter combinations
        lambda_grid, phi_grid, f_grid = np.meshgrid(lambda_range, phi_range, aspect_range, indexing='ij')
        
        # Reshape the grids into arrays
        lambda_values = lambda_grid.flatten()
        phi_values = phi_grid.flatten()
        f_values = f_grid.flatten()

    L1_istime = track_parameters['L1_ist_obs']
    L1_isspeed = track_parameters['L1_isv_obs']

    L1_istime = datetime.datetime.strptime(L1_istime, "%Y-%m-%d %H:%M")
    lead_time = np.round((L1_istime - endtime).total_seconds()/3600., 2)
    
    if np.isnan(L1_isspeed):
        L1_isspeed = None

    elon = tracks_interpolated
    time = timeaxis

    thi = time[0]

    if track_parameters['do_ensemble']:
        print('ELEvoHI in ensemble mode.')
        runnumber = 0
        ensemble = pd.DataFrame()
    else:
        runnumber = 0
        ensemble = pd.DataFrame()
        lambda_values = np.array([np.deg2rad(track_parameters['halfwidth'][0])])
        phi_values = np.array([np.deg2rad(track_parameters['phi_manual'][0])])
        f_values = np.array([f])

    # L1
    coord = get_horizons_coord('SEMB-L1', thi)
    sc_hee = coord.transform_to(frames.HeliocentricEarthEcliptic)  #HEE
    sc_heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
    L1_time = sc_heeq.obstime.to_datetime()
    L1_r = sc_heeq.radius.value
    L1_lon = np.deg2rad(sc_heeq.lon.value)
    L1_lat = np.deg2rad(sc_heeq.lat.value)

    # Earth
    coord = get_horizons_coord(399, thi)
    sc_hee = coord.transform_to(frames.HeliocentricEarthEcliptic)  #HEE
    sc_heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
    earth_time = sc_heeq.obstime.to_datetime()
    earth_r = sc_heeq.radius.value
    earth_lon = np.deg2rad(sc_heeq.lon.value)
    earth_lat = np.deg2rad(sc_heeq.lat.value)

    coord = get_horizons_coord('STEREO-A', thi)
    sc_hee = coord.transform_to(frames.HeliocentricEarthEcliptic)  #HEE
    sc_heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ

    sta_time = sc_heeq.obstime.to_datetime()
    sta_r = sc_heeq.radius.value
    sta_lon = np.deg2rad(sc_heeq.lon.value)
    sta_lat = np.deg2rad(sc_heeq.lat.value)
    
    # logging runnumbers for which no DBMfit converges
    nofit = []

    # variable is set to 1 in case no fit is possible for deterministic run
    no_det_run = False

    for halfwidth, phi, f in zip(lambda_values, phi_values, f_values):
        if track_parameters['do_ensemble']:
            print('Parameters for this ensemble member:')
            print('phi: ', round(np.rad2deg(phi)))
            print('halfwidth: ', round(np.rad2deg(halfwidth)))
            print('inverse ellipse aspect ratio: ', round(f, 1))
            runnumber = runnumber + 1
            print('runnumber: ', runnumber)
        else:
            print('ELEvoHI is in single mode.')

            print('phi: ', round(np.rad2deg(phi)))
            print('halfwidth: ', round(np.rad2deg(halfwidth)))
            print('inverse ellipse aspect ratio: ', round(f, 1))

            runnumber = runnumber + 1
            print('runnumber: ', runnumber)   

        # TODO: Implement for STEREO-B
        #STEREO Ahead
        if track_parameters['HIobs'] == 'A':
            d = sta_r
            if sta_lon >=0:
                direction = sta_lon - phi
            else:
                direction = sta_lon + phi      

        else:
            raise NotImplementedError("STEREO-B observations not implemented in this function.")
            #STEREO Behind
            # d = stb_r
            # if stb_lon >=0:
            #     direction = stb_lon - phi
            # else:
            #     direction = stb_lon + phi

        if abs(direction) + abs(earth_lon) < np.pi:
            delta_earth = direction - earth_lon
        else:
            delta_earth = direction - (earth_lon + 2 * np.pi * np.sign(direction))

        if abs(direction) + abs(L1_lon) < np.pi:
            delta_L1 = direction - L1_lon
        else:
            delta_L1 = direction - (L1_lon + 2 * np.pi * np.sign(direction))

        hit_L1 = does_cme_hit_elevohi(delta_L1, halfwidth)

        if not track_parameters['do_ensemble']:
            det_plot = True
            det_run_no = runnumber
        else:   
            if round(np.rad2deg(det_run[0])) == round(np.rad2deg(phi)) and round(det_run[1], 1) == round(f, 1) and round(np.rad2deg(det_run[2])) == round(np.rad2deg(halfwidth)):
                det_plot = True
                print('det_run set to True in ensemble!')
                det_run_no = runnumber

            else:
                det_plot = False

        elon_rad = np.deg2rad(elon)
        R_elcon = functions.ELCon(elon_rad, d, phi, halfwidth, f)   

        gamma_valid, winds_valid, res_valid, tinit, rinit, vinit, swspeed, xdata, ydata = functions.DBMfitting(time, R_elcon, prediction_path, det_plot, startfit=startcut, endfit=endcut)

        # check if DBMfit found at least one converging result
        if winds_valid[0] == 0:
            nofit.append(runnumber)
            continue

        # make equidistant grid for ELEvo times, with 10 min resolution
        start_time = tinit

        # Define time step as a timedelta object
        time_step = datetime.timedelta(minutes=10)

        # Define number of time steps
        timegrid = 1440

        # Generate time array
        time_array = [start_time + i * time_step for i in range(timegrid)]

        # convert to datetime
        #time_array = [tim.to_pydatetime() for tim in time_array]

        # create 1-D DBM kinematic for ellipse apex with
        # constant drag parameter and constant background solar wind speed

        gamma = gamma_valid[0]
        winds = winds_valid[0]

        # numeric time axis starting at zero
        tnum = [(time_array[i] - start_time).total_seconds() for i in range(timegrid)]
        # speed array
        vdrag = np.zeros(timegrid, dtype=float)
        # distance array
        rdrag = np.zeros(timegrid, dtype=float)

        # then use Vrsnak et al. 2013 equation 5 for v(t), 6 for r(t)

        # acceleration or deceleration
        # Note that the sign in the dbm equation is related to vinit and the ambient wind speed.

        if vinit < winds:
            accsign = -1
            print('negative')
        else:
            accsign = 1
            print('positive')

        for i in range(timegrid):
            # heliocentric distance of CME apex in km
            rdrag[i] = (accsign / (gamma)) * np.log(1 + (accsign * (gamma) * ((vinit - winds) * tnum[i]))) + winds * tnum[i] + rinit
            # speed in km/s
            vdrag[i] = (vinit - winds) / (1 + (accsign * (gamma) * ((vinit - winds) * tnum[i]))) + winds

            if not np.isfinite(rdrag[i]):
                print('Sign of gamma does not fit to vinit and w!')
                raise ValueError('Invalid value in rdrag')
        
        R = rdrag /AU_in_km

        prediction = functions.elevo_new(R, time_array, tnum, f, halfwidth, hit_L1, delta_L1, L1_r, L1_lon)
        
        if prediction is not None:
            target_l1_present = prediction['target'] == 'L1'

            if type(target_l1_present) == bool:
                target_l1_present = np.array([target_l1_present])

            if target_l1_present.any():
                dt_L1, dv_L1, prediction = functions.assess_prediction(prediction, 'L1', L1_istime, L1_isspeed)
                any_dt_present = True

            tmp_ensemble = pd.DataFrame()
            tmp_ensemble['run no.'] = [int(runnumber)] #* len(prediction)
            tmp_ensemble['target'] = prediction['target']
            tmp_ensemble['apex direction (HEE)'] = round(np.rad2deg(delta_earth))
            tmp_ensemble['phi [° from HI observer]'] = round(np.rad2deg(phi))
            tmp_ensemble['halfwidth [°]'] = round(np.rad2deg(halfwidth))
            tmp_ensemble['inv. aspect ratio'] = round(f, 1)
            tmp_ensemble['startcut'] = startcut
            tmp_ensemble['endcut'] = endcut
            tmp_ensemble['elongation min. [°]'] = round(elon[startcut], 1)
            tmp_ensemble['elongation max. [°]'] = round(elon[endcut], 1)
            tmp_ensemble['tinit [UT]'] = tinit.strftime("%Y-%m-%d %H:%M")
            tmp_ensemble['rinit [R_sun]'] = round(rinit/rsun_in_km)
            tmp_ensemble['vinit [km/s]'] = round(vinit)
            tmp_ensemble['drag parameter [e-7/km]'] = round(gamma*1e7, 2)
            tmp_ensemble['solar wind speed [km/s]'] = round(winds)
            tmp_ensemble['dec (+)/acc (-)'] = accsign
            tmp_ensemble['arrival time [UT]'] = prediction['arrival time [UT]']
            tmp_ensemble['arrival speed [km/s]'] = prediction['arrival speed [km/s]']
            tmp_ensemble['dt [h]'] = prediction['dt [h]']
            tmp_ensemble['dv [km/s]'] = prediction['dv [km/s]']
            tmp_ensemble['cmeID_elevo'] = track_parameters['cmeID_elevo']
            
            if lead_time:
                tmp_ensemble['prediction lead time [h]'] = lead_time
            else:
                tmp_ensemble['prediction lead time [h]'] = None
                
            ensemble = pd.concat([ensemble, tmp_ensemble])

            ensemble.loc[ensemble['target'].isna(), 'target'] = 'No hit!'           

            if det_plot:
                det_results = prediction
                
            plt.close('all')
        else:
            print('No prediction possible for run no.: ', runnumber)
            if det_plot:
                no_det_run = True
            tmp_ensemble = pd.DataFrame()
            ensemble = pd.concat([ensemble, tmp_ensemble])


    return ensemble


def run_elevohi_updated(track_times, track_elongs, track_parameters):

    AU_in_km, rsun_in_km = get_constants()

    phi = track_parameters['phi_manual'][0]
    halfwidth = track_parameters['halfwidth'][0]
    f = track_parameters['f'][0]

    starttime = datetime.datetime.strptime(track_times[0].strftime("%Y-%m-%d %H:%M"), "%Y-%m-%d %H:%M")
    endtime = datetime.datetime.strptime((track_times[-1]+ datetime.timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M"), "%Y-%m-%d %H:%M") 

    timeaxis, tracks_interpolated = interpolate_tracks(track_times, track_elongs, starttime=starttime, endtime=endtime)

    startcut = timeaxis.searchsorted(starttime)
    endcut = timeaxis.searchsorted(endtime) - 1

    if np.any(np.isnan(tracks_interpolated)):
        raise ValueError('NaN values in elongation track.')

    phi = np.deg2rad(phi)
    halfwidth = np.deg2rad(halfwidth)

    # Create a grid of parameter combinations
    lambda_values = np.array([np.deg2rad(track_parameters['halfwidth'][0])])
    phi_values = np.array([np.deg2rad(track_parameters['phi_manual'][0])])
    f_values = np.array([f])


    elon = tracks_interpolated
    time = timeaxis

    thi = time[0]
    time_num = [(t - thi).total_seconds() for t in time]
    runnumber = 0

    ## lets assume we dont look at data before stereoa was launched duh
    sta_time,sta_r,sta_lon,sta_lat = get_obj_heeq_pos(thi, 'STEREO-A')
    sta_time, sta_r, sta_lon, sta_lat = sta_time[0], sta_r[0], sta_lon[0], sta_lat[0]

    ## lets get L1 position etc.. for the first time of track
    L1_time,L1_r,L1_lon,L1_lat = get_obj_heeq_pos(thi, 'SEMB-L1')
    L1_time, L1_r, L1_lon, L1_lat = L1_time[0], L1_r[0], L1_lon[0], L1_lat[0]
    ensemble = pd.DataFrame()

    predictions = []

    det_run_no = 0

    R_Elcon_arr = []
    R_time_arr = []

    for halfwidth, phi, f in zip(lambda_values, phi_values, f_values):

        d = sta_r
        if sta_lon >=0:
            direction = sta_lon - phi

        else:
            direction = sta_lon + phi


        delta_target = compute_delta(L1_lon, direction)
        delta_sta    = compute_delta(sta_lon, direction)
        hit = does_cme_hit_elevohi(delta_target,halfwidth)

        elon_rad = np.deg2rad(elon)
        R_elcon = functions.ELCon(elon_rad, d, phi, halfwidth, f)   

        ## run DBMfitting    

        R_Elcon_arr.append(R_elcon)
        R_time_arr.append(time)

        gamma_valid, winds_valid, res_valid, tinit, rinit, vinit, swspeed, xdata, ydata = functions.DBMfitting(time,
                                                                                                             R_elcon,
                                                                                                             startfit=startcut,
                                                                                                             endfit=endcut,
                                                                                                             prediction_path=track_parameters['prediction_path'],
                                                                                                             det_plot=False,
                                                                                                             silent=1)

        if(winds_valid[0]==0):
            continue

        start_time = tinit

        # Define time step as a timedelta object
        time_step = datetime.timedelta(minutes=10)

        timegrid = 1440

        # Generate time array
        time_array = [start_time + i * time_step for i in range(timegrid)]
        
        # convert to datetime
        #time_array = [tim.to_pydatetime() for tim in time_array]

        # create 1-D DBM kinematic for ellipse apex with
        # constant drag parameter and constant background solar wind speed
        tnum = [(time_array[i] - start_time).total_seconds() for i in range(timegrid)]
        
        # create 1-D DBM kinematic for ellipse apex with
        # constant drag parameter and constant background solar wind speed

        gamma = gamma_valid[0]
        winds = winds_valid[0]

        # speed array
        vdrag = np.zeros(timegrid, dtype=float)
        # distance array
        rdrag = np.zeros(timegrid, dtype=float)

        if vinit < winds:
            accsign = -1
            #print('negative')
        else:
            accsign = 1
            #print('positive')

        for i in range(timegrid):
            # heliocentric distance of CME apex in km
            rdrag[i] = (accsign / (gamma)) * np.log(1 + (accsign * (gamma) * ((vinit - winds) * tnum[i]))) + winds * tnum[i] + rinit
            # speed in km/s
            vdrag[i] = (vinit - winds) / (1 + (accsign * (gamma) * ((vinit - winds) * tnum[i]))) + winds

            if not np.isfinite(rdrag[i]):
                print('Sign of gamma does not fit to vinit and w!')
                raise ValueError('Invalid value in rdrag')
        
        # ########################################################################
        # # run the final prediction using ELEvo   
        
        R = rdrag /AU_in_km

        if hit == False:
            pred = None

        else:
            # computed_arrival = functions.compute_arrival(R, vdrag, time_array, L1_r)
            # pred = {"target": "L1",
            #         "arrival time [UT]": computed_arrival['arr_time_fin'][0].replace(second=0, microsecond=0),
            #         "arrival speed [km/s]": int(round(computed_arrival['arr_speed_list'][0])),
            #         "dt [h]": np.nan,
            #         "dv [km/s]": np.nan}

            pred = functions.elevo_new(R, time_array, tnum, f, halfwidth, 1, delta_target, L1_r, L1_lon)


        if pred is not None:
            arrival_dt, arrival_dv, prediction = functions.assess_prediction(pred, 'L1', track_parameters['L1_ist_obs'], track_parameters['L1_isv_obs'])
            
            formatted_drag = "{:.2f}".format(gamma*1e7)

            predictions.append(pred)
            tmp_ensemble = pd.DataFrame()
            tmp_ensemble['cmeID_elevo'] = [track_parameters['cmeID_elevo'].replace(':','_').replace('-','_')]
            tmp_ensemble['cmeID_strudl'] = [track_parameters['cmeID_strudl']]
            tmp_ensemble['run no.'] = [int(det_run_no+1)]
            tmp_ensemble['target'] = "L1"
            tmp_ensemble['apex direction (HEE)'] = round(np.rad2deg(delta_target))
            tmp_ensemble['phi [° from HI observer]'] = round(np.rad2deg(phi))
            tmp_ensemble['halfwidth [°]'] = round(np.rad2deg(halfwidth))
            tmp_ensemble['inv. aspect ratio'] = round(f, 1)
            tmp_ensemble['startcut'] = 0
            tmp_ensemble['endcut'] = 30
            tmp_ensemble['elongation min. [°]'] = round(elon[0], 1)
            tmp_ensemble['elongation max. [°]'] = round(elon[-1], 1)
            tmp_ensemble['tinit [UT]'] = tinit.strftime("%Y-%m-%d %H:%M")
            tmp_ensemble['rinit [R_sun]'] = round(rinit/rsun_in_km)
            tmp_ensemble['vinit [km/s]'] = round(vinit)
            tmp_ensemble['drag parameter [e-7/km]'] = round(gamma*1e7, 2)
            tmp_ensemble['solar wind speed [km/s]'] = round(winds)
            tmp_ensemble['dec (+)/acc (-)'] = accsign
            tmp_ensemble['arrival time [UT]'] = prediction['arrival time [UT]']
            tmp_ensemble['arrival speed [km/s]'] = prediction['arrival speed [km/s]']
            tmp_ensemble['dt [h]'] = prediction['dt [h]']
            tmp_ensemble['dv [km/s]'] = prediction['dv [km/s]']
            ensemble = pd.concat([ensemble, tmp_ensemble])
            print('Prediction done for run no.: ', track_parameters['cmeID_strudl'])
        else:
            print('No prediction possible for run no.: ', track_parameters['cmeID_strudl'])

    return ensemble

def main(strudl_track_with_parameters_path, results_save_path, config=None, use_baseline=False):

    all_ensembles = pd.DataFrame()

    if config is None:
        tracks_times_science, tracks_elongs_science, parameters = load_strudl_tracks(strudl_track_with_parameters_path, return_parameters=True)
    else:
        event_path =  '/Users/maikebauer/Code/ELEvoHI_Python/data/timestep/'
        files = sorted([file for file in os.listdir(event_path) if file.endswith('.csv')])

        tracks_times_science = []
        tracks_elongs_science = []

        file_configs = []
        for file in files:
            df = pd.read_csv(event_path + file)
            tracks_times_science.append(pd.to_datetime(df['TRACK_DATE'].values))
            tracks_elongs_science.append(df['ELON'].values)
            file_configs.append(event_path + 'config/' + file.replace('.csv', '.json').replace('track_', 'config_'))

    if use_baseline:
        baseline_suffix = 'baseline'
    else:
        baseline_suffix = 'updated'

    no_pred_possible = 0

    for track_num, track in enumerate(tracks_times_science):

        prediction_path = results_save_path+str(track_num)+'_'

        if config is None:
            cmeID_elevo = parameters[track_num]['cmeID_elevo']
            cmeID_strudl = parameters[track_num]['cmeID_strudl']

            print('Processing STRUDL ID: ', cmeID_strudl, ' ELEvo ID: ', cmeID_elevo)

            use_vinit_donki_category = True

            if use_vinit_donki_category:
                vinit_donki_category = parameters[track_num]['vinit']
                if vinit_donki_category > 900:
                    vinit_donki_category = 'fast'
                else:
                    vinit_donki_category = 'slow'
            else:
                vinit_donki_category = None

            parameters_track = {
                        'halfwidth': [parameters[track_num]['halfwidth']]*3,
                        'halfwidth_step': np.nan,
                        'phi_manual': [parameters[track_num]['phi']]*3,
                        'phi_step': np.nan,
                        'f':[0.7, 0.7 ,0.7],
                        'f_step': np.nan,
                        'phi_FPF_range': np.nan,
                        'L1_ist_obs': parameters[track_num]['L1_ist_obs'],
                        "startcut": 'start',
                        "endcut": 'end',
                        "L1_isv_obs": 999,
                        "cmeID_elevo": cmeID_elevo,
                        "cmeID_strudl": cmeID_strudl,
                        "do_ensemble": False,
                        "phi_FPF": False,
                        "prediction_path": prediction_path,
                        "HIobs": 'A',
                        "vinit_donki_category": vinit_donki_category
                    }  

        else:
            parameters_track = functions.load_config(file_configs[track_num])
            parameters_track['cmeID_elevo'] = file_configs[track_num].split('/')[-1].replace('config_', '').replace('.json', '')
            print('Processing ELEvo ID: ', parameters_track['cmeID_elevo'])

        if use_baseline:
            ensemble = run_elevohi_baseline(track, tracks_elongs_science[track_num], parameters_track)
        else:
            ensemble = run_elevohi_updated(track, tracks_elongs_science[track_num], parameters_track)

        if ensemble.empty:
            print('No valid ensemble members for track number: ', track_num)
            no_pred_possible += 1
            continue

        all_ensembles = pd.concat([all_ensembles, ensemble])

        

    return all_ensembles, no_pred_possible, track_num+1


if __name__ == "__main__":
    
    raw = 'tracks_with_parameters_mean_45_2024_05_01_2025_04_30_earth_pa_6h_raw_2025_11_27.npy'
    corrected = 'tracks_with_parameters_mean_45_2024_05_01_2025_04_30_earth_pa_6h_cleaned_2025_11_27.npy'

    files = [raw, corrected]
    baselines = [True, False]
    results = []

    strudl_track_with_parameters_path = '/Users/maikebauer/CME_ML/Model_Train/run_25062025_120013_model_cnn3d/'+corrected
    results_save_path = '/Users/maikebauer/Code/ELEvoHI_Python/results/'

    for bnum, use_baseline in enumerate(baselines):
        if use_baseline:
            config = True
        else:
            config = None

        ensemble, no_pred, no_total = main(strudl_track_with_parameters_path=strudl_track_with_parameters_path,
            results_save_path=results_save_path,
            config=config,
            use_baseline=use_baseline)
                
        results.append({'dt [h]': ensemble['dt [h]'].abs().mean(),
                        'dv [km/s]': ensemble['dv [km/s]'].abs().mean(),
                        'no prediction possible': no_pred,
                        'total events': no_total,
                        'baseline': use_baseline})
        
    print('Results: ')
    for res in results:
        print(res)

