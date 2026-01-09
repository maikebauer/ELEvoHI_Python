from functions import *
from natsort import natsorted
import glob
from scipy import interpolate,optimize
import datetime
import os
from sunpy.coordinates import frames, get_horizons_coord
from sunpy.coordinates import Helioprojective, HeliocentricEarthEcliptic, HeliographicStonyhurst
from astropy.coordinates import SkyCoord

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
    print(pos_time)
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
    new_time_axis = calculate_new_time_axis(min_times, max_times, cadence=40)

    interpolated_elongs = []
    for t in range(0,len(track_times)):
        times = track_times[t]
        x     = [(times[i] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s') for i in range(0,len(times))]
        times = [(times[i]-min_times).total_seconds() for i in range(0,len(times))]
        interpf = interpolate.interp1d(x, track_elongs[t],kind='linear', fill_value="extrapolate")
        x_new = (new_time_axis - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        interpolated_elongs.append(interpf(x_new))
    

    return new_time_axis, np.array(interpolated_elongs).mean(0)

def interpolate_tracks(track_times,track_elongs):

    min_time    = min(track_times)
    max_time    = max(track_times)
    new_time_axis = calculate_new_time_axis(min_time, max_time, cadence=40)

    x     = [(track_times[i] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s') for i in range(0,len(track_times))]
    times = [(track_times[i]-min_time).total_seconds() for i in range(0,len(track_times))]
    interpf = interpolate.interp1d(x, track_elongs,kind='linear', fill_value="extrapolate")
    x_new = (new_time_axis - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    interpolated_elongs = interpf(x_new)
    

    return new_time_axis, np.array(interpolated_elongs)


def compute_delta(target_lon, direction):
    if abs(direction) + abs(target_lon) < np.pi:
        delta = direction - target_lon
    else:
        delta = direction - (target_lon + 2 * np.pi * np.sign(direction))

    # if np.abs(direction) + np.abs(target_lon) > np.pi and np.sign(direction) != np.sign(target_lon):
    #     print('First Case')
    #     delta = direction - (target_lon + 2 * np.pi * np.sign(direction))
    # else:
    #     print('Second Case')
    #     delta = direction - target_lon
    
    # delta_deg = np.round(np.rad2deg(delta),1)
    # print('computed delta (deg): ', delta_deg)
    # delta = np.deg2rad(delta_deg)
    
    return delta


def does_cme_hit(delta,halfwidth, tol=0.0):
    tol_rad = np.deg2rad(tol)

    return np.abs(delta) < (halfwidth + tol_rad)

def main(strudl_track_with_parameters_path, results_save_path, config=None, use_vinit_donki=False, use_baseline=False):
    
    if config is None:
        tracks_times_science, tracks_elongs_science, parameters = load_strudl_tracks(strudl_track_with_parameters_path, return_parameters=True)
    else:
        tracks_times_science, tracks_elongs_science = load_strudl_tracks(strudl_track_with_parameters_path, return_parameters=False)

    ensemble = pd.DataFrame()

    winds_valid_arr = []
    res_valid_arr = []
    R_Elcon_arr = []
    R_time_arr = []
    vinit_arr = []
    vinit_donki_arr = []
    cmeID_arr = []
    cmeID_strudl_arr = []
    invalid_arr = []
    
    valid_runs = 0
    total_runs = 0

    if use_baseline:
        baseline_suffix = 'baseline'
    else:
        baseline_suffix = 'updated'

    for track_num, track in enumerate(tracks_times_science):

        # if parameters[track_num]['cmeID_strudl'] != 'CME_13':
        #     continue
        total_runs +=1
        print('Number of total runs so far: ', total_runs)
        # timeaxis, tracks_interpolated = combine_tracks(track, tracks_elongs_science[track_num])
        timeaxis, tracks_interpolated = interpolate_tracks(track, tracks_elongs_science[track_num])
        print('Processing track number: ', track_num+1, ' out of ', len(tracks_times_science))
        print('Processing STRUDL ID: ', parameters[track_num]['cmeID_strudl'], ' ELEvo ID: ', parameters[track_num]['cmeID_elevo'])

        if np.any(np.isnan(tracks_interpolated)):
            print('Skipping track number ', track_num, ' due to NaN values in elongation track.')
            invalid_arr.append(track_num)
            continue
        
        if config is None:
            phi = parameters[track_num]['phi']
            halfwidth = parameters[track_num]['halfwidth']
            target_dt = parameters[track_num]['L1_ist_obs']
            cmeID_elevo = parameters[track_num]['cmeID_elevo']
            cmeID_strudl = parameters[track_num]['cmeID_strudl']
            aspect_range = np.array([0.7])

        else:
            phi = config['phi_manual'][0]
            halfwidth = config['halfwidth'][0]
            target_dt = datetime.strptime(config['L1_ist_obs'],'%Y-%m-%d %H:%M')

            num_points_f = int((config['f'][2] - config['f'][1])/config['f_step'] + 1)
            aspect_range = np.linspace(config['f'][1], config['f'][2], num_points_f)

        if use_vinit_donki:
            vinit_donki = parameters[track_num]['vinit']
            
        else:
            vinit_donki = None

        use_vinit_donki_category = True
        if use_vinit_donki_category:
            vinit_donki_category = parameters[track_num]['vinit']
            if vinit_donki_category > 900:
                vinit_donki_category = 'fast'
            else:
                vinit_donki_category = 'slow'
        else:
            vinit_donki_category = None
        # f_range = [config['f'][1], config['f'][2]]
        # f_step = config['f_step']

        # start_phi = np.deg2rad(phi - config['phi_FPF_range'])
        # end_phi = np.deg2rad(phi + config['phi_FPF_range'])
        # num_points_phi = int(round((np.rad2deg(end_phi) - np.rad2deg(start_phi))/np.rad2deg(p_step) + 1))

        phi = np.deg2rad(phi)
        halfwidth = np.deg2rad(halfwidth)

        ### create grid for ensemble parameters 
        # start_lambda = np.deg2rad(hw_range[0])
        # end_lambda = np.deg2rad(hw_range[1])
        # num_points_lambda = int((hw_range[1] - hw_range[0])/np.rad2deg(hw_step) + 1)

        # start_f = f_range[0]
        # end_f = f_range[1]
        # num_points_f = int((end_f - start_f)/f_step + 1)

        # lambda_range = np.linspace(start_lambda, end_lambda, num_points_lambda)
        # phi_range = np.linspace(start_phi, end_phi, num_points_phi)
        # aspect_range = np.linspace(start_f, end_f, num_points_f)

        # Create a grid of parameter combinations
        lambda_grid, phi_grid, f_grid = np.meshgrid(halfwidth, phi, aspect_range, indexing='ij')

        # Reshape the grids into arrays
        lambda_values = lambda_grid.flatten()
        phi_values = phi_grid.flatten()
        f_values = f_grid.flatten()

        elon = tracks_interpolated
        time = timeaxis

        thi = time[0]
        time_num = [(t - thi).total_seconds() for t in time]
        runnumber = 0

        ## lets assume we dont look at data before stereoa was launched duh
        sta_time,sta_r,sta_lon,sta_lat = get_obj_heeq_pos(track[0], 'STEREO-A')
        sta_time, sta_r, sta_lon, sta_lat = sta_time[0], sta_r[0], sta_lon[0], sta_lat[0]
        ## lets get L1 position etc.. for the first time of track
        L1_time,L1_r,L1_lon,L1_lat = get_obj_heeq_pos(track[0], 'SEMB-L1')
        L1_time, L1_r, L1_lon, L1_lat = L1_time[0], L1_r[0], L1_lon[0], L1_lat[0]
        # ensemble = pd.DataFrame()

        predictions = []

        det_run_no = 0
        for halfwidth, phi, f in zip(lambda_values, phi_values, f_values):

            d = sta_r
            if sta_lon >=0:
                direction = sta_lon - phi
                # direction_deg = np.round(np.rad2deg(sta_lon - phi), 0)
                # direction = np.deg2rad(direction_deg)#sta_lon - phi
            else:
                direction = sta_lon + phi
                # direction_deg = np.round(np.rad2deg(sta_lon + phi), 0)
                # direction = np.deg2rad(direction_deg)#sta_lon + phi

            delta_target = compute_delta(L1_lon, direction)
            delta_sta    = compute_delta(sta_lon, direction)
            hit = does_cme_hit(delta_target,halfwidth)

            elon_rad = np.deg2rad(elon)
            R_elcon = ELCon(elon_rad, d, phi, halfwidth, f)   

            ## run DBMfitting    

            R_Elcon_arr.append(R_elcon)
            R_time_arr.append(time)

            if use_baseline:
                gamma_valid, winds_valid, res_valid, tinit, rinit, vinit, swspeed, xdata, ydata = DBMfitting(time, R_elcon, startfit=0, endfit=30, prediction_path=results_save_path+str(track_num)+'_', det_plot=False,silent=1, vinit_donki=vinit_donki)

            else:
                #gamma_valid, winds_valid, res_valid, tinit, rinit, vinit, swspeed, xdata, ydata = DBMfitting_updated(time, R_elcon, startfit=0, endfit=30, prediction_path=results_save_path+str(track_num)+'_', det_plot=False,silent=1, vinit_donki=vinit_donki, vinit_donki_cat=vinit_donki_category)

                # Identify all candidate start points
                # Rsun_in_AU = 6.957e8 / AU   # 1 Rsun in AU
                # threshold  = 50 * Rsun_in_AU

                # start_indices = np.where(R_elcon < threshold)[0]

                # If nothing below 50 Rsun, use first point only.
                # if len(start_indices) == 0:
                #     start_indices = np.array([0])

                # TODO: Fix this
                start_indices = np.array([0])

                # Containers for aggregated fits
                all_gamma_valid = []
                all_winds_valid = []
                all_res_valid   = []
                all_tinit       = []
                all_rinit       = []
                all_vinit       = []
                all_swspeed     = []
                all_xdata       = []
                all_ydata       = []
                all_start_idx   = [] 


                for s in start_indices:

                    # Cut the track starting at s
                    time_cut = time[s:].copy()
                    R_cut    = R_elcon[s:].copy()

                    # Guard: need at least 3 points to fit
                    if len(time_cut) < 3:
                        continue

                    # Call DBM fitting
                    prediction_path=results_save_path+str(track_num)+'_'
                    
                    gamma_valid, winds_valid, res_valid, tinit, rinit, vinit, swspeed, xdata, ydata = DBMfitting_updated(
                                                                                                                            time_cut, R_cut,
                                                                                                                            startfit=0, endfit=30,
                                                                                                                            prediction_path=prediction_path,
                                                                                                                            det_plot=False,
                                                                                                                            silent=1,
                                                                                                                            vinit_donki=vinit_donki,
                                                                                                                            vinit_donki_cat=vinit_donki_category
                                                                                                                        )

                    # Skip if DBM did not return any valid fits
                    if not isinstance(gamma_valid, np.ndarray) and gamma_valid == 0:
                        continue

                    # function returns arrays/lists for gamma_valid, winds_valid, res_valid
                    for idx in range(len(gamma_valid)):

                        all_gamma_valid.append(gamma_valid[idx])
                        all_winds_valid.append(winds_valid[idx])
                        all_res_valid.append(res_valid[idx])

                        # These are single values per DBM-call, so just repeat for each solution
                        all_tinit.append(tinit)
                        all_rinit.append(rinit)
                        all_vinit.append(vinit)
                        all_swspeed.append(swspeed)

                        # store plotting data too
                        all_xdata.append(xdata)
                        all_ydata.append(ydata)

                        all_start_idx.append(s)  # optional tracking


                # If no fits succeeded at all:
                if len(all_gamma_valid) == 0:
                    gamma_valid = [0]
                    winds_valid = [0]
                    res_valid   = [np.nan]
                    tinit       = time[0]
                    rinit       = R_elcon[0]
                    vinit       = np.nan
                    swspeed     = np.nan
                    xdata       = []
                    ydata       = []

                else:
                    gamma_valid = all_gamma_valid
                    winds_valid = all_winds_valid
                    res_valid   = all_res_valid

                    tinit       = all_tinit[0]
                    rinit       = all_rinit[0]
                    vinit       = all_vinit[0]
                    swspeed     = all_swspeed[0]
                    xdata       = all_xdata[0]
                    ydata       = all_ydata[0]


            winds_valid_arr.append(winds_valid)
            res_valid_arr.append(res_valid)
            vinit_arr.append(vinit)
            vinit_donki_arr.append(parameters[track_num]['vinit'])
            cmeID_arr.append(cmeID_elevo)
            cmeID_strudl_arr.append(cmeID_strudl)

            if(winds_valid[0]==0):
                invalid_arr.append(track_num)
                continue

            start_time = tinit

            # Define time step as a timedelta object
            time_step = timedelta(minutes=10)

            timegrid = 1440

            # Generate time array
            time_array = [start_time + i * time_step for i in range(timegrid)]
            
            # convert to datetime
            time_array = [tim.to_pydatetime() for tim in time_array]

            # create 1-D DBM kinematic for ellipse apex with
            # constant drag parameter and constant background solar wind speed
            tnum = [(time_array[i] - start_time).total_seconds() for i in range(timegrid)]

            if use_baseline:
                gamma_valid = gamma_valid[0:1]

            else:
                # TODO: Fix this
                sorted_inds = np.argsort(res_valid)
                gamma_valid = np.array(gamma_valid)[sorted_inds]
                winds_valid = np.array(winds_valid)[sorted_inds]
                res_valid   = np.array(res_valid)[sorted_inds]

                gamma_valid = gamma_valid[0:1]

            for valid_ids in range(len(gamma_valid)):
                gamma = gamma_valid[valid_ids]
                winds = winds_valid[valid_ids]

                # numeric time axis starting at zero
                # speed array
                vdrag = np.zeros(timegrid, dtype=float)
                # distance array
                rdrag = np.zeros(timegrid, dtype=float)

                # then use Vrsnak et al. 2013 equation 5 for v(t), 6 for r(t)

                # acceleration or deceleration
                # Note that the sign in the dbm equation is related to vinit and the ambient wind speed.

                if vinit < winds:
                    accsign = -1
                    # print('negative')
                else:
                    accsign = 1
                    # print('positive')

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
                
                R = rdrag /AU

                if hit == False:
                    pred = None
                else:
                    #pred = elevo_new(R, time_array, tnum, f, halfwidth, hit,delta_target,L1_r,L1_lon)

                    computed_arrival = compute_arrival(R, vdrag, time_array, L1_r)
                    pred = {"target": "L1",
                            "arrival time [UT]": computed_arrival['arr_time_fin'][0].replace(second=0, microsecond=0),
                            "arrival speed [km/s]": int(round(computed_arrival['arr_speed_list'][0])),
                            "dt [h]": np.nan,
                            "dv [km/s]": np.nan}


                if pred is not None:
                    arrival_dt, arrival_dv, prediction = assess_prediction(pred, 'L1', target_dt, np.nan)
                
                    
                    pred['dt [h]'] = arrival_dt
                    pred['dv [km/s]'] = arrival_dv
                    formatted_drag = "{:.2f}".format(gamma*1e7)

                    predictions.append(pred)
                    tmp_ensemble = pd.DataFrame()
                    tmp_ensemble['cmeID_elevo'] = [parameters[track_num]['cmeID_elevo']]
                    tmp_ensemble['cmeID_strudl'] = [parameters[track_num]['cmeID_strudl']]
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
                    tmp_ensemble['rinit [R_sun]'] = round(rinit/rsun)
                    tmp_ensemble['vinit [km/s]'] = round(vinit)
                    tmp_ensemble['drag parameter [e-7/km]'] = gamma #round(gamma*1e7, 2)
                    tmp_ensemble['solar wind speed [km/s]'] = round(winds)
                    tmp_ensemble['dec (+)/acc (-)'] = accsign
                    tmp_ensemble['arrival time [UT]'] = pred['arrival time [UT]']
                    tmp_ensemble['arrival speed [km/s]'] = pred['arrival speed [km/s]']
                    tmp_ensemble['dt [h]'] = round((pred["arrival time [UT]"] - target_dt).total_seconds()/3600,2)
                    tmp_ensemble['dv [km/s]'] = pred['dv [km/s]']
                    ensemble = pd.concat([ensemble, tmp_ensemble])
                    print('Prediction done for run no.: ', parameters[track_num]['cmeID_strudl'])
                    valid_runs += 1
                else:
                    print('No prediction possible for run no.: ', parameters[track_num]['cmeID_strudl'])
                det_run_no+=1

    results_name = strudl_track_with_parameters_path.split('/')[-1].replace('tracks_with_parameters_','elevohi_results_').replace('.npy','_'+baseline_suffix+'.csv')
    ensemble.to_csv(results_save_path+ results_name, na_rep='NaN', index=False)

    elcon_dict = {
        'R_Elcon': R_Elcon_arr,
        'Time': R_time_arr,
        'vinit': vinit_donki_arr,
        'cmeID_elevo': cmeID_arr,
        'cmeID_strudl': cmeID_strudl_arr
    }
    elcon_save_name = strudl_track_with_parameters_path.split('/')[-1].replace('tracks_with_parameters_','elcon_tracks_').replace('.npy','_'+baseline_suffix+'.npy')
    np.save(results_save_path + elcon_save_name, elcon_dict)

    print('Invalid runs for track numbers: ', invalid_arr)
    print('Total valid runs: ', valid_runs)
    print('Total runs: ', total_runs)

if __name__ == "__main__":
    
    config = {
                'halfwidth': [25,15,35],
                'halfwidth_step': 5,
                'phi_manual': [64,60,68],
                'phi_step': 2,
                'f':[0.8, 0.7 ,1],
                'f_step': 0.1,
                'phi_FPF_range': 10,
                'L1_ist_obs': '2020-07-13 6:00',
                "startcut": 0,
                "endcut": 30,
                "L1_isv_obs": 345.94
            }   

    raw = 'tracks_with_parameters_mean_45_2024_05_01_2025_04_30_earth_pa_6h_raw_2025_11_27.npy'
    corrected = 'tracks_with_parameters_mean_45_2024_05_01_2025_04_30_earth_pa_6h_cleaned_2025_11_27.npy'

    files = [raw, corrected]
    baselines = [True, False]

    for file_type in files:
        for use_baseline in baselines:
            strudl_track_with_parameters_path = '/Users/maikebauer/CME_ML/Model_Train/run_25062025_120013_model_cnn3d/'+file_type
            results_save_path = '/Users/maikebauer/Code/ELEvoHI_Python/results/'

            main(strudl_track_with_parameters_path=strudl_track_with_parameters_path,
                results_save_path=results_save_path,
                config=None,
                use_vinit_donki=False,
                use_baseline=use_baseline)