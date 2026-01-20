import pytest
import pandas as pd
import datetime

@pytest.fixture
def load_test_post_conjunction_run_elevohi_baseline():
    
    post_conj_input_file = 'tests/fixtures/track_2024_07_27T06_36_00_CME_001.csv'

    post_conj_config = {
                        "basic_path": "/Users/maikebauer/Code/",
                        "track_path": "/Users/maikebauer/Code/ELEvoHI_Python/data",
                        "eventdate": "20240509",
                        "HIobs": "A",
                        "mode": "science",
                        "do_ensemble": False,
                        "phi_FPF": False,
                        "phi_FPF_range": [
                        10,
                        10
                        ],
                        "phi_manual": [
                        15.676488035380373,
                        15.676488035380373,
                        15.676488035380373
                        ],
                        "phi_step": 2,
                        "f": [
                        0.7,
                        0.7,
                        0.7
                        ],
                        "f_step": 0.1,
                        "halfwidth": [
                        42.,
                        42.,
                        42.
                        ],
                        "halfwidth_step": 5,
                        "starttime": datetime.datetime.strptime("2024-07-27 13:00", "%Y-%m-%d %H:%M"),
                        "endtime": datetime.datetime.strptime("2024-07-27 23:59", "%Y-%m-%d %H:%M"),
                        "outer_system": False,
                        "movie": False,
                        "silent": False,
                        "L1_ist_obs": datetime.datetime.strptime("2024-07-29 19:13", "%Y-%m-%d %H:%M"),
                        "L1_isv_obs": 999,
                        "cmeID_elevo": "2024_07_27T06_36_00_CME_001",
                        "vinit": None,
                        "prediction_path": ""
                        }

    tracks_times = []
    tracks_elongs = []

    df = pd.read_csv(post_conj_input_file)
    tracks_times.append(pd.to_datetime(df['TRACK_DATE'].values))
    tracks_elongs.append(df['ELON'].values)

    output_dt = 33.78
    output_dv = -632.

    return tracks_times, tracks_elongs, post_conj_config, output_dt, output_dv

@pytest.mark.parametrize("input", [
    'load_test_post_conjunction_run_elevohi_baseline'
])
def test_run_elevohi_baseline(input, request):
    tracks_times, tracks_elongs, post_conj_config, expected_dt, expected_dv = request.getfixturevalue(input)

    from code_base.run_elevohi import run_elevohi_new, make_implementation

    impl = make_implementation(use_baseline=True, setup_config=None)
    ensemble = run_elevohi_new(tracks_times[0], tracks_elongs[0], post_conj_config, impl)

    dt = ensemble[0]['dt [h]'].values[0]
    dv = ensemble[0]['dv [km/s]'].values[0]
    
    tol_hours = 0.001
    tol_kms = 0.1

    assert abs(dt - expected_dt) < tol_hours
    assert abs(dv - expected_dv) < tol_kms