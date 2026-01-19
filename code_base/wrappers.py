import code_base.functions as functions
import datetime

class DBMFit:
    def __init__(self,*,gamma, wind, tinit,rinit,vinit,residual,is_valid):

        self.gamma = gamma
        self.wind = wind
        self.tinit = tinit
        self.rinit = rinit
        self.vinit = vinit
        self.residual = residual
        self.is_valid = is_valid

    def get_parameters(self):
        return self.gamma, self.wind, self.tinit, self.rinit, self.vinit, self.residual, self.is_valid


class DBMResult:
    def __init__(self, fits_dict=None):
        self.fits = fits_dict if fits_dict else {}

    def add_fit(self, fit_id, gamma, wind, tinit, rinit, vinit, residual):
        self.fits[fit_id] = DBMFit(
            gamma=gamma,
            wind=wind,
            tinit=tinit,
            rinit=rinit,
            vinit=vinit,
            residual=residual,
            is_valid=True if wind != 0 else False
        )

    def compute_best_fit(self):
        best_fit = {}
        lowest_residual = float('inf')
        fit_id_best = None

        for fit_id, fit_data in self.fits.items():
            if fit_data.is_valid and fit_data.residual < lowest_residual:
                lowest_residual = fit_data.residual
                best_fit = fit_data
                fit_id_best = fit_id

        self.fits = {}
        if fit_id_best is not None:
            self.fits[fit_id_best] = best_fit

    
def run_dbm_fitting_wrapper(
    *,
    time,
    distance_au,
    startfit,
    endfit,
    prediction_path,
    det_plot,
    silent,
    vinit_input=None,
    setup_config=None
):
    """
    Unified DBM fitting interface for the pipeline.
    """

    use_updated = False
    use_vinit_donki_cat = None
    allow_multiple_dbm_fits = False

    if setup_config is not None:
        use_updated = setup_config.get("use_dbm_updated", False)
        use_vinit_donki_cat = setup_config.get("use_vinit_donki_category", False)
        allow_multiple_dbm_fits = setup_config.get("allow_multiple_dbm_fits", False)

    if use_updated:
        gamma_valid, winds_valid, res_valid, tinit, rinit, vinit, swspeed, xdata, ydata = functions.DBMfitting_updated(
            time=time,
            distance_au=distance_au,
            prediction_path=prediction_path,
            det_plot=det_plot,
            startfit=startfit,
            endfit=endfit,
            silent=silent,
            use_vinit_donki_cat=use_vinit_donki_cat,
            vinit_input=vinit_input
        )

    else:
        gamma_valid, winds_valid, res_valid, tinit, rinit, vinit, swspeed, xdata, ydata = functions.DBMfitting(
            time=time,
            distance_au=distance_au,
            prediction_path=prediction_path,
            det_plot=det_plot,
            startfit=startfit,
            endfit=endfit,
            silent=silent,
        )

    dbm_result = DBMResult()


    for i in range(len(gamma_valid)):
        fit_id = i
        dbm_result.add_fit(
            fit_id=fit_id,
            gamma=gamma_valid[i],
            wind=winds_valid[i],
            tinit=tinit,
            rinit=rinit,
            vinit=vinit,
            residual=res_valid[i]
        )
    
    if not allow_multiple_dbm_fits:
        dbm_result.compute_best_fit()
    
    return dbm_result

def run_cme_hit_function_wrapper(
    *,
    delta,
    halfwidth,
    setup_config=None):

    """
    Unified CME hit function interface for the pipeline.
    """

    use_updated = False

    if setup_config is not None:
        use_updated = setup_config.get("use_cme_hit_function_updated", False)

    if use_updated:
        return functions.does_cme_hit(
            delta=delta,
            halfwidth=halfwidth
        )

    else:
        return functions.does_cme_hit_elevohi(
            delta=delta,
            halfwidth=halfwidth
        )
    
def run_arrival_computation_wrapper(
    *,
    R,
    vdrag,
    time_array,
    L1_r,
    tnum,
    f,
    halfwidth,
    delta_target,
    L1_lon,
    setup_config=None
):

    """
    Unified arrival time computation interface for the pipeline.
    """

    use_updated = False

    if setup_config is not None:
        use_updated = setup_config.get("use_arrival_computation_updated", False)

    if use_updated:
        pred = functions.compute_arrival_wrapper(R, vdrag, time_array, L1_r)

    else:
        pred = functions.elevo_new(R, time_array, tnum, f, halfwidth, 1, delta_target, L1_r, L1_lon)

    return pred

class BaselineImplementation:

    def cme_hit(self, *, delta, halfwidth):
        return run_cme_hit_function_wrapper(
            delta=delta,
            halfwidth=halfwidth,
            setup_config=None
        )

    def dbm_fit(self, *, time, distance_au, startcut, endcut, prediction_path, vinit):
        return run_dbm_fitting_wrapper(
            time=time,
            distance_au=distance_au,
            startfit=startcut,
            endfit=endcut,
            prediction_path=prediction_path,
            det_plot=False,
            silent=1,
            vinit_input=None,
            setup_config=None
        )

    def arrival(self, *, R, vdrag, time_array, tnum, f, halfwidth, delta_target, L1_r, L1_lon):
        return run_arrival_computation_wrapper(
            R=R,
            vdrag=vdrag,
            time_array=time_array,
            L1_r=L1_r,
            tnum=tnum,
            f=f,
            halfwidth=halfwidth,
            delta_target=delta_target,
            L1_lon=L1_lon,
            setup_config=None
        )
    
class UpdatedImplementation:

    def __init__(self, setup_config):
        self.setup_config = setup_config

    def cme_hit(self, *, delta, halfwidth):
        return run_cme_hit_function_wrapper(
            delta=delta,
            halfwidth=halfwidth,
            setup_config=self.setup_config
        )

    def dbm_fit(self, *, time, distance_au, startcut, endcut, prediction_path, vinit):
        return run_dbm_fitting_wrapper(
            time=time,
            distance_au=distance_au,
            startfit=startcut,
            endfit=endcut,
            prediction_path=prediction_path,
            det_plot=False,
            silent=1,
            vinit_input=vinit,
            setup_config=self.setup_config
        )

    def arrival(self, *, R, vdrag, time_array, tnum, f, halfwidth, delta_target, L1_r, L1_lon):
        return run_arrival_computation_wrapper(
            R=R,
            vdrag=vdrag,
            time_array=time_array,
            L1_r=L1_r,
            tnum=tnum,
            f=f,
            halfwidth=halfwidth,
            delta_target=delta_target,
            L1_lon=L1_lon,
            setup_config=self.setup_config
        )

