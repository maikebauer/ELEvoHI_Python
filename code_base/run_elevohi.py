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
import code_base.wrappers as wrappers
from dataclasses import dataclass

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
        elongs.append(np.array(list(converted_strudl_dict[cme_key]["elongation"])))

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
        min_time = min(track_times).replace(second=0, microsecond=0)
    else:
        min_time = starttime

    if endtime is None:
        max_time = max(track_times).replace(second=0, microsecond=0)
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

def get_constants():
    AU_in_km = const.au.to_value('km')
    rsun_in_km = const.R_sun.to_value('km')

    return AU_in_km, rsun_in_km

def run_elevohi_new(track_times, track_elongs, params, implementation_obj):
    """
    Unified ELEvoHI.
    """
    AU_in_km, rsun_in_km = get_constants()

    phi = np.deg2rad(params["phi_manual"][0])
    halfwidth = np.deg2rad(params["halfwidth"][0])
    f = params["f"][0]

    L1_ist_obs = params["L1_ist_obs"]
    L1_isv_obs = params.get("L1_isv_obs", None)

    prediction_path = params["prediction_path"]
    vinit = params['vinit']
    cmeID_elevo = params["cmeID_elevo"]
    cmeID_strudl = params.get("cmeID_strudl", None)

    if params.get("starttime", None) is None:
        starttime = track_times[0].replace(second=0, microsecond=0)
    else:
        starttime = params["starttime"]

    if params.get("endtime", None) is None:
        endtime = track_times[-1].replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
    else:
        endtime = params["endtime"]

    timeaxis, elon_interp = interpolate_tracks(track_times, track_elongs, starttime=starttime, endtime=endtime)

    startcut = timeaxis.searchsorted(starttime)
    endcut = timeaxis.searchsorted(endtime) - 1

    if np.any(np.isnan(elon_interp)):
        raise ValueError("NaN values in interpolated elongation track")

    elon_rad = np.deg2rad(elon_interp)
    time = timeaxis
    thi = time[0]

    sta_time, sta_r, sta_lon, _ = get_obj_heeq_pos(thi, "STEREO-A")
    sta_r = sta_r[0]
    sta_lon = sta_lon[0]

    L1_time, L1_r, L1_lon, _ = get_obj_heeq_pos(thi, "SEMB-L1")
    L1_r = L1_r[0]
    L1_lon = L1_lon[0]

    if sta_lon >= 0:
        direction = sta_lon - phi
    else:
        direction = sta_lon + phi

    delta_target = compute_delta(L1_lon, direction)

    hit = implementation_obj.cme_hit(delta=delta_target, halfwidth=halfwidth)

    R_elcon = functions.ELCon(elon_rad, sta_r, phi, halfwidth, f)

    dbm_result = implementation_obj.dbm_fit(
                                            time=time,
                                            distance_au=R_elcon,
                                            startcut=startcut,
                                            endcut=endcut,
                                            prediction_path=prediction_path
                                        )

    dfs = []

    for fit_id, fit in dbm_result.fits.items():
        if fit.is_valid:
            gamma, winds, tinit, rinit, vinit, residual, is_valid = fit.get_parameters()
            
        else:
            dfs.append(pd.DataFrame())  # no valid fit
            continue
        
        # TODO: Insert elevo ensemble option here
        # should take gamma, wind, timesteps, rinit and vinit from dbm fit
        # should return arrays of rdrag and vdrag for the given time steps
        # gamma, winds, and vinit should be part of a distribution as in elevo ensembles
        # should be single values for regular elevohi run
        R, vdrag, time_array, tnum = implementation_obj.dbm_kinematics(
            gamma=gamma,
            winds=winds,
            tinit=tinit,
            rinit=rinit,
            vinit=vinit
        )
        # R, vdrag, time_array, tnum = functions.compute_dbm_kinematics_single(tinit, rinit, vinit, gamma, winds)
        # ends here

        pred = None
        if hit:
            pred, cme_distance_r = implementation_obj.arrival(
                R=R,
                vdrag=vdrag,
                time_array=time_array,
                tnum=tnum,
                f=f,
                halfwidth=halfwidth,
                delta_target=delta_target,
                L1_r=L1_r,
                L1_lon=L1_lon
            )

        if pred is None:
            dfs.append(pd.DataFrame())
            continue

        _, _, prediction = functions.assess_prediction(
            pred,
            "L1",
            L1_ist_obs,
            L1_isv_obs
        )

        df = pd.DataFrame({
            "cmeID_elevo": [cmeID_elevo],
            "cmeID_strudl": [cmeID_strudl],
            "target": ["L1"],
            "phi [°]": [round(np.rad2deg(phi))],
            "halfwidth [°]": [round(np.rad2deg(halfwidth))],
            "inv. aspect ratio": [round(f, 2)],
            "tinit [UT]": [tinit.strftime("%Y-%m-%d %H:%M")],
            "rinit [R_sun]": [round(rinit / rsun_in_km, 2)],
            "vinit [km/s]": [round(vinit)],
            "drag parameter [e-7/km]": [round(gamma * 1e7, 2)],
            "solar wind speed [km/s]": [round(winds)],
            "arrival time [UT]": prediction["arrival time [UT]"],
            "arrival speed [km/s]": prediction["arrival speed [km/s]"],
            "dt [h]": prediction["dt [h]"],
            "dv [km/s]": prediction["dv [km/s]"],
        })

        dfs.append(df)
    
    return dfs

def main_new(strudl_track_with_parameters_path, results_save_path, impl=None, config=None):

    all_ensembles = pd.DataFrame()
    no_pred_possible = 0

    if config is None:
        tracks_times_science, tracks_elongs_science, parameters = load_strudl_tracks(
            strudl_track_with_parameters_path,
            return_parameters=True
        )

    else:
        event_path = "/Users/maikebauer/Code/ELEvoHI_Python/data/timestep/"
        files = sorted([f for f in os.listdir(event_path) if f.endswith(".csv")])

        tracks_times_science = []
        tracks_elongs_science = []
        file_configs = []

        for file in files:
            df = pd.read_csv(event_path + file)
            tracks_times_science.append(pd.to_datetime(df["TRACK_DATE"].values))
            tracks_elongs_science.append(df["ELON"].values)
            file_configs.append(
                event_path + "config/" +
                file.replace(".csv", ".json").replace("track_", "config_")
            )


    for track_num, track in enumerate(tracks_times_science):

        prediction_path = results_save_path + f"{track_num}_"

        if config is None:
            cmeID_elevo = parameters[track_num]["cmeID_elevo"]
            cmeID_strudl = parameters[track_num]["cmeID_strudl"]

            print("Processing STRUDL ID:", cmeID_strudl,
                  "ELEvo ID:", cmeID_elevo)

            parameters_track = {
                "halfwidth": [parameters[track_num]["halfwidth"]] * 3,
                "halfwidth_step": np.nan,
                "phi_manual": [parameters[track_num]["phi"]] * 3,
                "phi_step": np.nan,
                "f": [0.7, 0.7, 0.7],
                "f_step": np.nan,
                "phi_FPF_range": np.nan,
                "L1_ist_obs": parameters[track_num]["L1_ist_obs"],
                "L1_isv_obs": 999,
                "cmeID_elevo": cmeID_elevo,
                "cmeID_strudl": cmeID_strudl,
                "do_ensemble": False,
                "phi_FPF": False,
                "prediction_path": prediction_path,
                "HIobs": "A",
                "vinit": parameters[track_num]["vinit"],
            }

        else:
            parameters_track = functions.load_config(file_configs[track_num])
            parameters_track["cmeID_elevo"] = file_configs[track_num].split("/")[-1].replace("config_", "").replace(".json", "")
            parameters_track["prediction_path"] = prediction_path
            parameters_track["vinit"] = None
            parameters_track["L1_ist_obs"] = datetime.datetime.strptime(parameters_track["L1_ist_obs"], "%Y-%m-%d %H:%M")
            parameters_track['starttime'] = datetime.datetime.strptime(parameters_track['starttime'], "%Y-%m-%d %H:%M")
            parameters_track['endtime'] = datetime.datetime.strptime(parameters_track['endtime'], "%Y-%m-%d %H:%M")

        ensemble = run_elevohi_new(
            track_times=track,
            track_elongs=tracks_elongs_science[track_num],
            params=parameters_track,
            implementation_obj=impl,
        )
        valid_ensemble = False
        for ens in ensemble:
            if ens.empty:
                continue
            else:
                valid_ensemble = True
                all_ensembles = pd.concat([all_ensembles, ens])

        if not valid_ensemble:
            print("No valid ensemble members for track number:", track_num)
            no_pred_possible += 1
            continue

    return all_ensembles, no_pred_possible, track_num + 1

def make_implementation(use_baseline, setup_config):
    if use_baseline:
        return wrappers.BaselineImplementation()
    
    return wrappers.UpdatedImplementation(setup_config)

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
            setup_config = None

        else:
            config = None

            setup_config = {
                "use_dbm_updated": False,
                "updated_dbm_vinit_computation": 'first',
                "allow_multiple_dbm_fits": False,
                "use_cme_hit_function_updated": False,
                "use_arrival_computation_updated": False,
                "use_elevo_ensembles": False,
                "kwargs_dbm": {"num_points": 3}
            }
        
        impl = make_implementation(use_baseline, setup_config)

        ensemble, no_pred, no_total = main_new(strudl_track_with_parameters_path=strudl_track_with_parameters_path,
            results_save_path=results_save_path,
            impl=impl,
            config=config)

        results.append({'dt [h]': ensemble['dt [h]'].abs().mean(),
                        'dv [km/s]': ensemble['dv [km/s]'].abs().mean(),
                        'no prediction possible': no_pred,
                        'total events': no_total,
                        'baseline': use_baseline})
        
    print('Results: ')
    for res in results:
        print(res)

