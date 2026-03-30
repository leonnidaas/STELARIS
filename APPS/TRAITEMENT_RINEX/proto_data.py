#!/usr/bin/env python
# coding: utf-8

# virer les futurewarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
from datetime import timezone
import georinex as gr
from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.navdata.operations import sort, concat
from gnss_lib_py.parsers.rinex_nav import _compute_eccentric_anomaly
from gnss_lib_py.parsers.rinex_nav import RinexNav
import gnss_lib_py.utils.constants as consts
from gnss_lib_py.utils.coordinates import add_el_az, ecef_to_geodetic
from gnss_lib_py.utils.time_conversions import datetime_to_gps_millis, gps_millis_to_datetime, gps_millis_to_tow
from gnss_lib_py.utils.gnss_models import _calculate_tropo_delay, _calculate_iono_delay
import numba as nb
import numpy as np
import pandas as pd
from time import time
from tqdm.auto import tqdm
import logging

FORMAT = "%(asctime)s %(module)s %(levelname)s %(message)s"
DATEFORMAT = "%Y-%m-%d %H:%M:%S"
# logging.basicConfig(format=FORMAT, datefmt=DATEFORMAT, level=logging.INFO)
# logging.getLogger("numba").setLevel(logging.WARNING)
# logging.getLogger("pandas").setLevel(logging.WARNING)
# logger = logging.getLogger(__name__)

# Example of parameters
# path_obs = "./SC01/rinex.obs"
# path_nav = "./SC01/rinex.nav"
# -i1 "./SC01/rinex.obs" -i2 "./SC01/rinex.nav"

# --- SYSTÈME DE FILTRE "RADICAL" (INTERCEPTE TOUT) ---
class SatErrorFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.sat_stats = {}

    def filter(self, record):
        # On intercepte le message peu importe le logger
        msg = record.getMessage()
        if "malformed line for" in msg:
            try:
                # On extrait le nom du satellite (ex: E34)
                sat_id = msg.split()[-1]
                self.sat_stats[sat_id] = self.sat_stats.get(sat_id, 0) + 1
            except:
                pass
            return False  # BLOQUE l'affichage dans le terminal
        return True

sat_filter = SatErrorFilter()

# On configure le format de base
logging.basicConfig(format=FORMAT, datefmt=DATEFORMAT, level=logging.INFO)

# ON APPLIQUE LE FILTRE À LA RACINE (ROOT)
# Cela force TOUS les loggers du projet à passer par ce filtre

logging.getLogger().addFilter(sat_filter)

# On réduit aussi le bruit des autres bibliothèques
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("pandas").setLevel(logging.WARNING)
logging.getLogger("georinex").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def print_sat_error_report():
    if sat_filter.sat_stats:
        print("\n" + "="*50)
        print("RÉSUMÉ DES ANOMALIES DE NAVIGATION (RINEX .nav)")
        print(f"{'Satellite':<15} | {'Points temporels impactés':<25}")
        print("-" * 50)
        for sat, count in sorted(sat_filter.sat_stats.items()):
            print(f"{sat:<15} | {count:<25}")
        print("="*50 + "\n")
    else:
        logger.info("Aucune anomalie détectée dans le fichier de navigation.")



class RinexObs_perso(NavData):
    """Class handling Rinex observation files [1]_.

    The Rinex Observation files (of the format .yyo) contain measured
    pseudoranges, carrier phase, doppler and signal-to-noise ratio
    measurements for multiple constellations and bands.
    This loader converts those file types into a NavData in which
    measurements from different bands are treated as separate measurement
    instances. Inherits from NavData().

    This class has primarily been built with Rinex v3.05 in mind but it
    should also work for prior Rinex versions.


    References
    ----------
    .. [1] https://files.igs.org/pub/data/format/rinex305.pdf


    """
    def __init__(self, xdata, obs_measure_types):
        """Loading Rinex observation files into a NavData based class.

        Should input path to `.yyo` file.

        Parameters
        ----------
        xdata : xarray

        obs_measure_types : dict

        """

        obs_file = xdata.to_dataframe()
        rx_bands = []
        for rx_measures in obs_measure_types.values():
            for single_measure in rx_measures:
                band = single_measure[1:]
                if band not in rx_bands:
                    rx_bands.append(band)
        obs_file.dropna(how='all', inplace=True)
        obs_file.reset_index(inplace=True)
        # Convert time to gps_millis
        datetime_series = [d.to_pydatetime().replace(tzinfo=timezone.utc)
                           for d in obs_file["time"]]
        obs_file['gps_millis'] = datetime_to_gps_millis(datetime_series)
        obs_file = obs_file.drop(columns=['time'])
        obs_file = obs_file.rename(columns={"sv":"sv_id"})
        # Convert gnss_sv_id to gnss_id and sv_id (plus gnss_sv_id)
        obs_navdata_raw = NavData(pandas_df=obs_file)
        obs_navdata_raw['gnss_sv_id'] = obs_navdata_raw['sv_id']
        gnss_chars = [sv_id[0] for sv_id in np.atleast_1d(obs_navdata_raw['sv_id'])]
        gnss_nums = [sv_id[1:] for sv_id in np.atleast_1d(obs_navdata_raw['sv_id'])]
        gnss_id = [consts.CONSTELLATION_CHARS[gnss_char] for gnss_char in gnss_chars]
        obs_navdata_raw['gnss_id'] = np.asarray(gnss_id)
        obs_navdata_raw['sv_id'] = np.asarray(gnss_nums, dtype=int)
        # Convert the coded column names to glp standards and extract information
        # into glp row and columns format
        info_rows = ['gps_millis', 'gnss_sv_id', 'sv_id', 'gnss_id']
        super().__init__()
        for band in rx_bands:
            rename_map = {}
            keep_rows = info_rows.copy()
            measure_type_dict = self._measure_type_dict()
            for measure_char, measure_row in measure_type_dict.items():
                measure_band_row = measure_char + band
                rename_map[measure_band_row] = measure_row
                keep_rows.append(measure_char + band)
                if measure_band_row not in obs_navdata_raw.rows:
                    obs_navdata_raw[measure_band_row] = np.array(len(obs_navdata_raw)*[np.nan])
            band_navdata = obs_navdata_raw.copy(rows=keep_rows)
            band_navdata.rename(rename_map, inplace=True)
            # Remove the cases with NaNs in the measurements
            nan_indexes = np.argwhere(np.isnan(band_navdata[["carrier_phase",
            	             "raw_doppler_hz",
                             "cn0_dbhz"]]).all(axis=0))[:,0].tolist()
            # Remove the cases with NaNs in the pseudorange
            nan_indexes += np.argwhere(np.isnan(band_navdata[["raw_pr_m"
                                                             ]]))[:,0].tolist()
            nan_indexes = sorted(list(set(nan_indexes)))
            if len(nan_indexes) > 0:
                band_navdata.remove(cols=nan_indexes,inplace=True)

            # Assign the gnss_lib_py standard names for signal_type
            rx_constellations = np.unique(band_navdata['gnss_id'])
            signal_type_dict = self._signal_type_dict()
            signal_types = np.empty(len(band_navdata), dtype=object)
            observation_codes = np.empty(len(band_navdata), dtype=object)
            for constellation in rx_constellations:
                signal_type = signal_type_dict[constellation][band]
                signal_types[band_navdata['gnss_id']==constellation] = signal_type
                observation_codes[band_navdata['gnss_id']==constellation] = band
            band_navdata['signal_type'] = signal_types
            band_navdata['observation_code'] = observation_codes
            if len(self) == 0:
                concat_navdata = concat(self, band_navdata)
            else:
                concat_navdata = concat(self, band_navdata)

            self.array = concat_navdata.array
            self.map = concat_navdata.map
            self.str_map = concat_navdata.str_map
            self.orig_dtypes = concat_navdata.orig_dtypes.copy()
            
        sort(self,'gps_millis', inplace=True)

    @staticmethod
    def _measure_type_dict():
        """Map of Rinex observation measurement types to standard names.

        Returns
        -------
        measure_type_dict : Dict
            Dictionary of the form {rinex_character : measure_name}
        """

        measure_type_dict = {'C': 'raw_pr_m',
                             'L': 'carrier_phase',
                             'D': 'raw_doppler_hz',
                             'S': 'cn0_dbhz'}
        return measure_type_dict

    @staticmethod
    def _signal_type_dict():
        """Dictionary from constellation and signal bands to signal types.

        Transformations from Section 5.1 in [2]_ and 5.2.17 from [3]_.

        Returns
        -------
        signal_type_dict : Dict
            Dictionary of the form {constellation_band : {band : signal_type}}

        References
        ----------
        .. [2] https://files.igs.org/pub/data/format/rinex304.pdf
        .. [3] https://files.igs.org/pub/data/format/rinex305.pdf

        """
        signal_type_dict = {}
        signal_type_dict['gps'] = {'1C' : 'l1',
                                   '1S' : 'l1',
                                   '1L' : 'l1',
                                   '1X' : 'l1',
                                   '1P' : 'l1',
                                   '1W' : 'l1',
                                   '1Y' : 'l1',
                                   '1M' : 'l1',
                                   '1N' : 'l1',
                                   '2C' : 'l2',
                                   '2D' : 'l2',
                                   '2S' : 'l2',
                                   '2L' : 'l2',
                                   '2X' : 'l2',
                                   '2P' : 'l2',
                                   '2W' : 'l2',
                                   '2Y' : 'l2',
                                   '2M' : 'l2',
                                   '2N' : 'l2',
                                   '5I' : 'l5',
                                   '5Q' : 'l5',
                                   '5X' : 'l5',
                                   }
        signal_type_dict['glonass'] = {'1C' : 'g1',
                                       '1P' : 'g1',
                                       '4A' : 'g1a',
                                       '4B' : 'g1a',
                                       '4X' : 'g1a',
                                       '2C' : 'g2',
                                       '2P' : 'g2',
                                       '6A' : 'g2a',
                                       '6B' : 'g2a',
                                       '6X' : 'g2a',
                                       '3I' : 'g3',
                                       '3Q' : 'g3',
                                       '3X' : 'g3',
                                       }
        signal_type_dict['galileo'] = {'1A' : 'e1',
                                       '1B' : 'e1',
                                       '1C' : 'e1',
                                       '1X' : 'e1',
                                       '1Z' : 'e1',
                                       '5I' : 'e5a',
                                       '5Q' : 'e5a',
                                       '5X' : 'e5a',
                                       '7I' : 'e5b',
                                       '7Q' : 'e5b',
                                       '7X' : 'e5b',
                                       '8I' : 'e5',
                                       '8Q' : 'e5',
                                       '8X' : 'e5',
                                       '6A' : 'e6',
                                       '6B' : 'e6',
                                       '6C' : 'e6',
                                       '6X' : 'e6',
                                       '6Z' : 'e6',
                                       }
        signal_type_dict['sbas'] = {'1C' : 'l1',
                                    '5I' : 'l5',
                                    '5Q' : 'l5',
                                    '5X' : 'l5',
                                    }
        signal_type_dict['qzss'] = {'1C' : 'l1',
                                    '1S' : 'l1',
                                    '1L' : 'l1',
                                    '1X' : 'l1',
                                    '1Z' : 'l1',
                                    '1B' : 'l1',
                                    '2S' : 'l2',
                                    '2L' : 'l2',
                                    '2X' : 'l2',
                                    '5I' : 'l5',
                                    '5Q' : 'l5',
                                    '5X' : 'l5',
                                    '5D' : 'l5',
                                    '5P' : 'l5',
                                    '5Z' : 'l5',
                                    '6S' : 'l6',
                                    '6L' : 'l6',
                                    '6X' : 'l6',
                                    '6E' : 'l6',
                                    '6Z' : 'l6',
                                    }
        signal_type_dict['beidou'] = {'2I' : 'b1',
                                      '2Q' : 'b1',
                                      '2X' : 'b1',
                                      '1D' : 'b1c',
                                      '1P' : 'b1c',
                                      '1X' : 'b1c',
                                      '1A' : 'b1c',
                                      '1N' : 'b1c',
                                      '1S' : 'b1a',
                                      '1L' : 'b1a',
                                      '1Z' : 'b1a',
                                      '5D' : 'b2a',
                                      '5P' : 'b2a',
                                      '5X' : 'b2a',
                                      '7I' : 'b2b',
                                      '7Q' : 'b2b',
                                      '7X' : 'b2b',
                                      '7D' : 'b2b',
                                      '7P' : 'b2b',
                                      '7Z' : 'b2b',
                                      '8D' : 'b2',
                                      '8P' : 'b2',
                                      '8X' : 'b2',
                                      '6I' : 'b3',
                                      '6Q' : 'b3',
                                      '6X' : 'b3',
                                      '6A' : 'b3',
                                      '6D' : 'b3a',
                                      '6P' : 'b3a',
                                      '6Z' : 'b3a',
                                      }
        signal_type_dict['irnss'] = {'5A' : 'l5',
                                     '5B' : 'l5',
                                     '5C' : 'l5',
                                     '5X' : 'l5',
                                     '9A' : 's',
                                     '9B' : 's',
                                     '9C' : 's',
                                     '9X' : 's',
                                     }
        return signal_type_dict


def _estimate_sv_clock_corr_perso(gps_millis, ephem, verbose=False):
    """Calculate the modelled satellite clock delay

    Parameters
    ---------
    gps_millis : int
        Time at which measurements are needed, measured in milliseconds
        since start of GPS epoch [ms].
    ephem : gnss_lib_py.parsers.navdata.NavData
        Satellite ephemeris parameters for measurement SVs.

    Returns
    -------
    clock_corr : np.ndarray
        Satellite clock corrections containing all terms [m].
    corr_polynomial : np.ndarray
        Polynomial clock perturbation terms [m].
    clock_relativistic : np.ndarray
        Relativistic clock correction terms [m].

    """
    # Extract required GPS constants
    ecc        = ephem['e']     # eccentricity
    sqrt_sma = ephem['sqrtA'] # sqrt of semi-major axis


    # if np.abs(delta_t).any() > 302400:
    #     delta_t = delta_t - np.sign(delta_t)*604800

    gps_week, gps_tow = gps_millis_to_tow(gps_millis)

    # Compute Eccentric Anomaly
    ecc_anom = _compute_eccentric_anomaly(gps_week, gps_tow, ephem)


    # Determine pseudorange corrections due to satellite clock corrections.
    # Calculate time offset from satellite reference time
    t_offset = gps_tow - ephem['t_oc']
        
    logger.debug(f"ecc : {ecc}")
    logger.debug(f"sqrt_sma : {sqrt_sma}")
    logger.debug(f"ecc_anom : {ecc_anom}")
    logger.debug(f"gps_tow - ephem['t_oc'] : {gps_tow} - {ephem['t_oc']}")
    logger.debug(f"t_offset : {t_offset}")
        
        
    if np.abs(t_offset).any() > 302400:  # pragma: no cover
        t_offset = t_offset-np.sign(t_offset)*604800
    #
    # Calculate clock corrections from the polynomial corrections in
    # broadcast message
    corr_polynomial = (ephem['SVclockBias']
                     + ephem['SVclockDrift']*t_offset
                     + ephem['SVclockDriftRate']*t_offset**2)
    
    # Calcualte the relativistic clock correction
    corr_relativistic = consts.F * ecc * sqrt_sma * np.sin(ecc_anom)
    
    # Calculate the total clock correction including the Tgd term
    ephem_TGD = np.nan_to_num(ephem['TGD'])
    ephem_BGD = np.nan_to_num(ephem['BGDe5a'])/(1.60200E9/1.24600E9 - 1)
    #clk_corr = (corr_polynomial - ephem_TGD + corr_relativistic)
    clk_corr = (corr_polynomial - ephem_TGD - ephem_BGD + corr_relativistic)
    
    logger.debug(f"ephem['SVclockBias'] : {ephem['SVclockBias']}")
    logger.debug(f"ephem['SVclockDrift'] : {ephem['SVclockDrift']}")
    logger.debug(f"ephem['SVclockDriftRate'] : {ephem['SVclockDriftRate']}")
    logger.debug(f"t_offset (bis) : {t_offset}")
    logger.debug(f"ephem['TGD'] : {ephem['TGD']}")
    logger.debug(f"corr_polynomial : {corr_polynomial}")
    logger.debug(f"corr_relativistic : {corr_relativistic}")
    logger.debug(f"clk_corr : {clk_corr}")

    #Convert values to equivalent meters from seconds
    clk_corr = np.array(consts.C*clk_corr, ndmin=1)
    corr_polynomial = np.array(consts.C*corr_polynomial, ndmin=1)
    corr_relativistic = np.array(consts.C*corr_relativistic, ndmin=1)

    return clk_corr, corr_polynomial, corr_relativistic


def find_sv_states_perso(gps_millis, ephem, verbose=False):
    """Compute position and velocities for all satellites in ephemeris file
    given time of clock.

    `ephem` contains broadcast ephemeris parameters (similar in form to GPS
    broadcast parameters).

    Must contain the following rows (description in [1]_):
    * :code:`gnss_id`
    * :code:`sv_id`
    * :code:`gps_week`
    * :code:`t_oe`
    * :code:`e`
    * :code:`omega`
    * :code:`Omega_0`
    * :code:`OmegaDot`
    * :code:`sqrtA`
    * :code:`deltaN`
    * :code:`IDOT`
    * :code:`i_0`
    * :code:`C_is`
    * :code:`C_ic`
    * :code:`C_rs`
    * :code:`C_rc`
    * :code:`C_uc`
    * :code:`C_us`

    Parameters
    ----------
    gps_millis : int
        Time at which measurements are needed, measured in milliseconds
        since start of GPS epoch [ms].
    ephem : gnss_lib_py.parsers.navdata.NavData
        NavData instance containing ephemeris parameters of satellites
        for which states are required.

    Returns
    -------
    sv_posvel : gnss_lib_py.parsers.navdata.NavData
        NavData containing satellite positions, velocities, corresponding
        time with GNSS ID and SV number.

    Notes
    -----
    Based on code written by J. Makela.
    AE 456, Global Navigation Sat Systems, University of Illinois
    Urbana-Champaign. Fall 2017

    More details on the algorithm used to compute satellite positions
    from broadcast navigation message can be found in [1]_.

    Satellite velocity calculations based on algorithms introduced in [2]_.

    References
    ----------
    ..  [1] Misra, P. and Enge, P,
        "Global Positioning System: Signals, Measurements, and Performance."
        2nd Edition, Ganga-Jamuna Press, 2006.
    ..  [2] B. F. Thompson, S. W. Lewis, S. A. Brown, and T. M. Scott,
        “Computing GPS satellite velocity and acceleration from the broadcast
        navigation message,” NAVIGATION, vol. 66, no. 4, pp. 769–779, Dec. 2019,
        doi: 10.1002/navi.342.

    """

    # Convert time from GPS millis to TOW
    gps_week, gps_tow = gps_millis_to_tow(gps_millis)
    # Extract parameters

    c_is = ephem['C_is']
    c_ic = ephem['C_ic']
    c_rs = ephem['C_rs']
    c_rc = ephem['C_rc']
    c_uc = ephem['C_uc']
    c_us = ephem['C_us']
    delta_n   = ephem['deltaN']

    ecc        = ephem['e']     # eccentricity
    omega    = ephem['omega'] # argument of perigee
    omega_0  = ephem['Omega_0']
    sqrt_sma = ephem['sqrtA'] # sqrt of semi-major axis
    sma      = sqrt_sma**2      # semi-major axis

    sqrt_mu_a = np.sqrt(consts.MU_EARTH) * sqrt_sma**-3 # mean angular motion
    gpsweek_diff = (np.mod(gps_week,1024) - np.mod(ephem['gps_week'],1024))*604800.
    sv_posvel = NavData()
    sv_posvel['gnss_id'] = ephem['gnss_id']
    sv_posvel['sv_id'] = ephem['sv_id']
    # Deal with times being a single value or a vector with the same
    # length as the ephemeris
    sv_posvel['gps_millis'] = gps_millis

    delta_t = gps_tow - ephem['t_oe'] + gpsweek_diff
    
    # Calculate the mean anomaly with corrections
    ecc_anom = _compute_eccentric_anomaly(gps_week, gps_tow, ephem)
    
    logger.debug(f"ephem GPS : {ephem['gnss_sv_id']}")
    logger.debug(f"gps_tow : {gps_tow}")
    logger.debug(f"ephem['t_oe'] : {ephem['t_oe']}")
    logger.debug(f"delta_t : {delta_t}")
    logger.debug(f"ecc_anom : {ecc_anom}")

    cos_e   = np.cos(ecc_anom)
    sin_e   = np.sin(ecc_anom)
    e_cos_e = (1 - ecc*cos_e)

    # Calculate the true anomaly from the eccentric anomaly
    sin_nu = np.sqrt(1 - ecc**2) * (sin_e/e_cos_e)
    cos_nu = (cos_e-ecc) / e_cos_e
    nu_rad     = np.arctan2(sin_nu, cos_nu)

    # Calcualte the argument of latitude iteratively
    phi_0 = nu_rad + omega
    phi   = phi_0
    for incl in range(5):
        cos_to_phi = np.cos(2.*phi)
        sin_to_phi = np.sin(2.*phi)
        phi_corr = c_uc * cos_to_phi + c_us * sin_to_phi
        phi = phi_0 + phi_corr

    # Calculate the longitude of ascending node with correction
    omega_corr = ephem['OmegaDot'] * delta_t
    
    logger.debug(f"omega_corr : {omega_corr}")

    # Also correct for the rotation since the beginning of the GPS week for which the Omega0 is
    # defined.  Correct for GPS week rollovers.

    # Also correct for the rotation since the beginning of the GPS week for
    # which the Omega0 is defined.  Correct for GPS week rollovers.
    omega = omega_0 - (consts.OMEGA_E_DOT*(gps_tow + gpsweek_diff)) + omega_corr

    # Calculate orbital radius with correction
    r_corr = c_rc * cos_to_phi + c_rs * sin_to_phi
    orb_radius      = sma*e_cos_e + r_corr

    ############################################
    ######  Lines added for velocity (1)  ######
    ############################################
    delta_e   = (sqrt_mu_a + delta_n) / e_cos_e
    dphi = np.sqrt(1 - ecc**2)*delta_e / e_cos_e
    # Changed from the paper
    delta_r   = (sma * ecc * delta_e * sin_e) + 2*(c_rs*cos_to_phi - c_rc*sin_to_phi)*dphi

    # Calculate the inclination with correction
    i_corr = c_ic*cos_to_phi + c_is*sin_to_phi + ephem['IDOT']*delta_t
    incl = ephem['i_0'] + i_corr

    logger.debug(f"delta_r : {delta_r}")
    logger.debug(f"i_corr : {i_corr}")

    ############################################
    ######  Lines added for velocity (2)  ######
    ############################################
    delta_i = 2*(c_is*cos_to_phi - c_ic*sin_to_phi)*dphi + ephem['IDOT']

    # Find the position in the orbital plane
    x_plane = orb_radius*np.cos(phi)
    y_plane = orb_radius*np.sin(phi)

    ############################################
    ######  Lines added for velocity (3)  ######
    ############################################
    delta_u = (1 + 2*(c_us * cos_to_phi - c_uc*sin_to_phi))*dphi
    dxp = delta_r*np.cos(phi) - orb_radius*np.sin(phi)*delta_u
    dyp = delta_r*np.sin(phi) + orb_radius*np.cos(phi)*delta_u
    # Find satellite position in ECEF coordinates
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)
    cos_i = np.cos(incl)
    sin_i = np.sin(incl)

    sv_posvel['x_sv_m'] = x_plane*cos_omega - y_plane*cos_i*sin_omega
    sv_posvel['y_sv_m'] = x_plane*sin_omega + y_plane*cos_i*cos_omega
    sv_posvel['z_sv_m'] = y_plane*sin_i

    ############################################
    ######  Lines added for velocity (4)  ######
    ############################################
    omega_dot = ephem['OmegaDot'] - consts.OMEGA_E_DOT
    logger.debug(f"omega_dot : {omega_dot}")
    
    sv_posvel['vx_sv_mps'] = (dxp * cos_omega
                         - dyp * cos_i*sin_omega
                         + y_plane  * sin_omega*sin_i*delta_i
                         - (x_plane * sin_omega + y_plane*cos_i*cos_omega)*omega_dot)

    sv_posvel['vy_sv_mps'] = (dxp * sin_omega
                         + dyp * cos_i * cos_omega
                         - y_plane  * sin_i * cos_omega * delta_i
                         + (x_plane * cos_omega - (y_plane*cos_i*sin_omega)) * omega_dot)

    sv_posvel['vz_sv_mps'] = dyp*sin_i + y_plane*cos_i*delta_i

    # Estimate SV clock corrections, including polynomial and relativistic
    # clock corrections
    clock_corr, _, _ = _estimate_sv_clock_corr_perso(gps_millis, ephem, verbose)

    sv_posvel['b_sv_m'] = clock_corr

    return sv_posvel


def extract_meas_and_nav(measurement_df, navigation_df, t):
    l_local_meas = []
    l_local_nav = []
    diff = np.abs(navigation_df['gps_millis'] - t)
    indices = np.arange(len(navigation_df))
    for i, sat in enumerate(measurement_df['gnss_sv_id']):
        index = navigation_df['gnss_sv_id'] == sat
        list_index = indices[index]
        if len(list_index) > 0:
            min = list_index[np.argmin(diff[index])]
            l_local_meas.append(i)
            l_local_nav.append(min)
        else:
            pass
    return l_local_meas, l_local_nav

logger.info("Compilating function residue_vect")
@nb.njit(nb.float64[:](nb.float64[:,:], nb.float64[:], nb.float64[:]))
def residue_vect(X, theta, pseudor):
    
    #est_pseudor = np.linalg.norm(X - theta[:3], axis=1) # Not working with numba
    est_pseudor = np.sqrt(np.sum((X - theta[:3])**2, axis=1))
    
    return est_pseudor + theta[3] - pseudor

logger.info("Compilating function grad_r_vect")
@nb.njit(nb.float64[:,:](nb.float64[:,:], nb.float64[:]))
def grad_r_vect(X, theta):
    
    N = len(X)
    J = np.ones((N,4))
    for i in range(N):
        J[i,:3] = (theta[:3] - X[i])/np.linalg.norm(theta[:3] - X[i])
        
    return J

logger.info("Compilating function compute_cov")
@nb.njit(nb.float64[:,:](nb.float64[:], nb.float64[:,:], nb.float64[:], nb.float64, nb.float64))
def compute_cov(theta, X, pranges, omega_e_dot=0.0, c_light=3e8):
    
    X_k = np.copy(X)
    
    # Sagnac or Earth-rotation correction
    d_theta = - omega_e_dot*(pranges - theta[3])/c_light # warning : we will apply the rotation in the opposite direction
    
    X_k[:,0] = np.cos(d_theta)*X[:,0] - np.sin(d_theta)*X[:,1]
    X_k[:,1] = np.sin(d_theta)*X[:,0] + np.cos(d_theta)*X[:,1]
    
    J = np.ascontiguousarray(grad_r_vect(X, theta))
    G = np.linalg.inv(np.dot(J.T,J))
    
    return G

logger.info("Compilating function optim")
@nb.njit(nb.float64[:](nb.float64[:,:], nb.float64[:], nb.float64, nb.float64, nb.int32, nb.float64))
def optim(X, pranges, omega_e_dot=0.0, c_light=3e8, n_iter_max=25, tol=1e-3):

    theta_k = np.zeros(4)
    delta_theta = np.ones(4)*np.inf
    force_loop = True
    X_k = np.copy(X)
    
    k = 0
    # careful
    # on first loop, np.linalg.norm(delta_theta) returns NaN on some systems
    while k < n_iter_max and (force_loop or np.linalg.norm(delta_theta) > tol):
        # after first iteration, disable force_loop
        force_loop = False
        # Sagnac or Earth-rotation correction
        d_theta = - omega_e_dot*(pranges - theta_k[3])/c_light # warning : we will apply the rotation in the opposite direction
        
        X_k[:,0] = np.cos(d_theta)*X[:,0] - np.sin(d_theta)*X[:,1]
        X_k[:,1] = np.sin(d_theta)*X[:,0] + np.cos(d_theta)*X[:,1]
        
        res_k = np.ascontiguousarray(residue_vect(X_k, theta_k, pranges)) # for Numba
        J_k = np.ascontiguousarray(grad_r_vect(X_k, theta_k)) # for Numba
        #J_k = grad_r_vect(X, theta_k)
        delta_theta = - np.linalg.inv(J_k.T @ J_k) @ J_k.T @ res_k
        theta_k += delta_theta
        
        k += 1
        
    return theta_k


def process_gnss_estim(obs_df, constellations, nav_df, iono_params):
    
    sv_states_all_time = []
    wls_solution = [] # final position estimate
    
    i = 0
    
    list_t = list(set(obs_df['gps_millis']))
    list_t.sort()
    
    pbar = tqdm(total=len(list_t))
    
    for t in list_t:
    
        measure_frame_df = obs_df[obs_df['gps_millis'] == t]
        measure_frame_df = measure_frame_df[np.isin(measure_frame_df['gnss_id'], constellations)]
        l_local_meas, l_local_nav = extract_meas_and_nav(measure_frame_df, nav_df, t)
        measure_frame_df = measure_frame_df.iloc[l_local_meas]
        ephem_df = nav_df.iloc[l_local_nav]
        ephem = NavData(pandas_df=ephem_df)
        measure_frame = NavData(pandas_df=measure_frame_df)
        
        pbar.update(1)
        
        # Sort the satellites
        rx_ephem = ephem
    
        # Modif Quentin
        transmission_time_gpst = measure_frame['gps_millis'] - measure_frame['raw_pr_m']/consts.C*1000
        
        if rx_ephem.shape[1] != measure_frame.shape[1]: #pragma: no cover
            raise RuntimeError('Some ephemeris data is missing')
    
        sv_states = find_sv_states_perso(transmission_time_gpst, rx_ephem)
        
        # Add them to new rows
        for row in sv_states.rows:
            if row not in ('gps_millis','gnss_id','sv_id'):
                measure_frame[row] = sv_states[row]

        # Estimation with L1 and E1 frequency bands only (single frequency estimation)
        indic_l1 = measure_frame['observation_code'] == '1C'
    
        if np.sum(indic_l1) >= 4: # minimum 4 satellites
    
            try:
                # Adjustment with iono and tropo
                measure_frame['corr_pr_m'] = measure_frame['raw_pr_m'] + measure_frame['b_sv_m']
                
                X = np.vstack([measure_frame['x_sv_m'], 
                       measure_frame['y_sv_m'], 
                       measure_frame['z_sv_m']]).T
                pseudor = measure_frame['corr_pr_m']
    
                # Selection l1/e1 band
                X = X[indic_l1,:]
                pseudor = pseudor[indic_l1]
                
                theta_i = optim(X, pseudor, omega_e_dot=consts.OMEGA_E_DOT, c_light=consts.C, n_iter_max=25, tol=1e-3) 
                #theta_geodetic = ecef_to_geodetic(theta_i[:3].reshape(1,-1))[0] # lat, lon, h
                state_wls_i = NavData()
                state_wls_i['gps_millis'] =  np.array(t)
                state_wls_i['x_rx_wls_m'] =  np.array(theta_i[0])
                state_wls_i['y_rx_wls_m'] =  np.array(theta_i[1])
                state_wls_i['z_rx_wls_m'] =  np.array(theta_i[2])
                state_wls_i['b_rx_wls_m'] =  np.array(theta_i[3])
                
                #state_wls_i = glp.solve_wls(measure_frame)
                measure_frame = add_el_az(measure_frame, state_wls_i, inplace=True)
        
                rx_ecef = state_wls_i[['x_rx_wls_m', 'y_rx_wls_m', 'z_rx_wls_m']]
        
                iono_corr = _calculate_iono_delay(measure_frame['gps_millis'], iono_params=iono_params, 
                                              rx_ecef=rx_ecef, sv_posvel=sv_states, constellation="gps")
                tropo_corr = _calculate_tropo_delay(measure_frame['gps_millis'], rx_ecef=rx_ecef, 
                                                sv_posvel=sv_states)
        
                measure_frame['iono_corr'] = iono_corr
                measure_frame['tropo_corr'] = tropo_corr
                measure_frame['corr_pr_m'] = measure_frame['raw_pr_m'] + measure_frame['b_sv_m'] - iono_corr - tropo_corr
            
                # Solution after update of information
                transmission_time_gpst_bis = measure_frame['gps_millis'] - measure_frame['corr_pr_m']/consts.C*1000
                
                #sv_states_bis = find_sv_states(transmission_time_gpst_bis, rx_ephem)
                sv_states_bis = find_sv_states_perso(transmission_time_gpst_bis, rx_ephem)
                
                #sort(sv_states_bis,ind=inv_sort_order,inplace=True) # !!! dangerous to use with extract_meas_and_nav
                for row in sv_states_bis.rows:
                    if row not in ('gps_millis','gnss_id','sv_id'):
                        measure_frame[row] = sv_states_bis[row]
                
                indic_l1_bis = measure_frame['observation_code'] == '1C'
                X = np.vstack([measure_frame['x_sv_m'], 
                               measure_frame['y_sv_m'], 
                               measure_frame['z_sv_m']]).T
                pseudor = measure_frame['corr_pr_m']
                # Selection l1/e1 band
                X = X[indic_l1_bis,:]
                pseudor = pseudor[indic_l1_bis]
                theta_i = optim(X, pseudor, omega_e_dot=consts.OMEGA_E_DOT, c_light=consts.C, n_iter_max=25, tol=1e-3)
                G = compute_cov(theta_i, X, pseudor, omega_e_dot=consts.OMEGA_E_DOT, c_light=consts.C)
                theta_geodetic = ecef_to_geodetic(theta_i[:3].reshape(1,-1))[0] # lat, lon, h
                state_wls_i_bis = NavData()
                state_wls_i_bis['gps_millis'] =  np.array(t)
                state_wls_i_bis['x_rx_wls_m'] =  np.array(theta_i[0])
                state_wls_i_bis['y_rx_wls_m'] =  np.array(theta_i[1])
                state_wls_i_bis['z_rx_wls_m'] =  np.array(theta_i[2])
                state_wls_i_bis['b_rx_wls_m'] =  np.array(theta_i[3])
                state_wls_i_bis['lat_rx_wls_deg'] =  np.array(theta_geodetic[0])
                state_wls_i_bis['lon_rx_wls_deg'] =  np.array(theta_geodetic[1])
                state_wls_i_bis['alt_rx_wls_m'] =  np.array(theta_geodetic[2])
                if np.sum(np.diag(G)[:3]) >= 0:   
                    state_wls_i_bis['pdop'] = np.sqrt(np.sum(np.diag(G)[:3]))
                else:
                    state_wls_i_bis['pdop'] = np.nan
                    logger.warning(f"Invalid PDOP with time {t}")
                
                wls_solution.append(state_wls_i_bis.pandas_df())
    
            except Exception as e:
                logger.error(f"Issue with time {t}")
        
        sv_states_all_time.append(measure_frame.pandas_df())
    
        i += 1
    
    pbar.close()

    sv_states_all_time = pd.concat(sv_states_all_time, ignore_index=True)
    sv_states_all_time['time'] = gps_millis_to_datetime(sv_states_all_time['gps_millis'])

    wls_solution = pd.concat(wls_solution, ignore_index=True)
    wls_solution['time_utc'] = gps_millis_to_datetime(wls_solution['gps_millis'])

    return sv_states_all_time, wls_solution


def process_rinex_files(input1: str, input2: str, output1=None, output2=None, verbose=True):
    """
    Process RINEX observation and navigation files.
    
    Parameters
    ----------
    input1 : str
        Path to the RINEX Observation file
    input2 : str
        Path to the RINEX Navigation file
    output1 : str, optional
        Path to store sv_state results (default: None, no output file)
    output2 : str, optional
        Path to store wls_solution results (default: None, no output file)
    verbose : bool, optional
        Enable verbose logging (default: True)
    
    Returns
    -------
    tuple
        (sv_states_all_time, wls_solution) as pandas DataFrames
    
    Raises
    ------
    FileNotFoundError
        If input files don't exist
    ValueError
        If RINEX files are invalid
    """
    path_obs = input1
    path_nav = input2

    # Rinex file reading
    if verbose:
        logger.info(f"Reading observation file {path_obs}")
    obs_header = gr.rinexheader(path_obs)
    if verbose:
        logger.info(f"Reading navigation file {path_nav}")
    nav_header = gr.rinexheader(path_nav)

    if verbose:
        logger.info("Reading of RINEX files")
    
    meas = ['C1C','L1C','S1C','C2L','L2L','S2L','C5Q','L5Q','S5Q']
    constellations = ['gps', 'galileo']
    obs_measure_types = {'G' : meas, 'E' : meas}

    t_start = time()
    obs = gr.load(path_obs, use=['G', 'E'], meas=meas, verbose=verbose)
    #obs = gr.load(path_obs, use=['G'], meas=meas, verbose=verbose, tlim=(date_start, date_end))
    #obs = gr.load(path_obs, use=['G', 'E'], meas=meas, verbose=verbose, tlim=(date_start, date_end))
    t_end = time()

    if verbose:
        logger.info(f"Total time = {(t_end - t_start):.1f} s, time per epoch = {((t_end - t_start)/len(obs.time)):.1f} s")

    # Observations

    # In 2021, UTC = GPS - 18 seconds
    #obs['time'] = obs['time'] - 18_000_000_000
    if 'LEAP SECONDS' in obs_header:
        leap_seconds_str = obs_header['LEAP SECONDS'].replace(' ','')
        if verbose:
            logger.info(f"Leap of {leap_seconds_str} seconds")
        leap_seconds = int(float(obs_header['LEAP SECONDS']) * 1_000_000_000)
        obs['time'] = obs['time'] - leap_seconds # requirement of GNSS_lib : datetimes of xarray in UTC time

    df_obs = RinexObs_perso(obs, obs_measure_types).pandas_df()


    # Navigation
    nav = RinexNav(path_nav)
    iono_params = nav.iono_params
    iono_params = iono_params[list(iono_params.keys())[0]]
    df_nav = nav.pandas_df()

    # Data processing
    if verbose:
        logger.info("Processing RINEX files")
    sv_states_all_time, wls_solution = process_gnss_estim(df_obs, constellations, df_nav, iono_params)

    # For storage
    if output1:
        if verbose:
            logger.info(f"Storing sv_states to {output1}")
        sv_states_all_time.to_csv(output1, index=False)

    if output2:
        if verbose:
            logger.info(f"Storing wls_solution to {output2}")
        wls_solution.to_csv(output2, index=False)
    
    print_sat_error_report()
    return sv_states_all_time, wls_solution


def get_parser()->'argparse.ArgumentParser':
    """Create argument parser for CLI usage."""
    parser = argparse.ArgumentParser(description="Process RINEX files.")
    parser.add_argument("-i1", "--input1", required=True, help="Path to the RINEX Observation file")
    parser.add_argument("-i2", "--input2", required=True, help="Path to the RINEX Navigation file")
    parser.add_argument("-o1", "--output1", required=False, default=None, help="Path to store sv_state results")
    parser.add_argument("-o2", "--output2", required=False, default=None, help="Path to the wls_solution results")
    parser.add_argument("-q", "--quiet", action="store_true", help="Disable verbose logging")
    return parser


def main():
    """Main entry point for CLI usage."""
    args = get_parser().parse_args()
    
    try:
        process_rinex_files(
            input1=args.input1,
            input2=args.input2,
            output1=args.output1,
            output2=args.output2,
            verbose=not args.quiet
        )
        print_sat_error_report()
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid data: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
