#!/usr/bin/env python
# coding: utf-8

# virer les futurewarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.neighbors import BallTree
from tqdm.auto import tqdm

import os
os.environ["KERAS_BACKEND"] = "torch"

import torch
import keras
import xgboost as xgb
import logging

from config import load_config_from_file
from models import get_keras_rnn_model, get_scaler_param, get_xgb_model, get_classes_param

FORMAT = "%(asctime)s %(module)s %(levelname)s %(message)s"
DATEFORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFORMAT, level=logging.INFO)
logging.getLogger("numpy").setLevel(logging.WARNING)
logging.getLogger("pandas").setLevel(logging.WARNING)
logging.getLogger("keras").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# Example of parameters
# -i "./SC01/groundtruth.csv"

# Parameters

list_prime = ['build','open-sky_rural','tree']
num_classes = len(list_prime)

stats = ['mean','std']
#stats=['mean','std', 'skew', 'kurtosis']

#feature_names = ['NSV', 'EL mean', 'EL std', 'EL skew', 'EL kurtosis', 'pdop', 'CN0 std', 'CN0 skew', 'CN0 kurtosis', 'CMC_l1']
feature_names = ['NSV', 'EL mean', 'EL std', 'pdop', 'CN0 std', 'CMC_l1']

sequence_length = 5


# Auxiliary stats functions



def euler_form(a, b):
    """
    Returns euler form of a complex number. 
    
    Parameters
    ----------
    a : float64
        Real part of the complex number. 
    b : float64
        Imaginary part of the complex number. 
        
    Returns
    -------
    R : float64
        Modulus of the complex number a+ib
    theta : float64
        Argument of the complex number a+ib (radian)
    """
    R = np.sqrt(a**2 + b**2)
    theta = np.arctan2(b,a)
    
    return R, theta


def moments(thetas, order):
    """
    Returns the k-th moment of an angle vector in Eulerian form. 
    
    Parameters
    ----------
    thetas : array-like of shape (n_samples,)
        An angle vector (degree).
        
    order : int
        Order of the moment you want to compute. 
        
    Returns
    -------
    R : float64
        Modulus of k-th moment. 
    theta : float64
        Argument of the k-th moment. 
    """
    assert isinstance(order, int) == True, "Only positive integer order"
    assert order > 0, "Only positive integer orders"
    
    if order == 1:
        a = np.mean(np.cos(order*thetas))
        b = np.mean(np.sin(order*thetas))
        return euler_form(a, b)
    else:
        R_1, theta_1 = moments(thetas, order=1)
        a = np.mean(np.cos(order*(thetas - theta_1)))
        b = np.mean(np.sin(order*(thetas - theta_1)))    
        R_p, theta_p = euler_form(a, b)
        return R_1, theta_1, R_p, theta_p
    
    R = np.sqrt(a**2 + b**2)
    theta = np.arctan2(b,a)
    
    return R, theta


def circular_mean(thetas):
    """
    Returns the circular mean of an angle vector.
    
    Parameters
    ----------
    thetas : array-like of shape (n_samples,)
        An angle vector (degree). 
    
    Returns
    -------
    circ_mean : float64
        The circular mean of the vector taken as input (radian).  
    """
    thetas = np.radians(thetas) # degree to radian
    
    l_cos = np.cos(thetas)
    l_sin = np.sin(thetas)
    
    cos_mean = np.nanmean(l_cos)
    sin_mean = np.nanmean(l_sin)
    
    if cos_mean>=0:
        circ_mean = np.arctan(sin_mean/cos_mean)
    else:
        circ_mean = np.arctan(sin_mean/cos_mean)+np.pi
        
    return circ_mean


def circular_dispersion(thetas):
    """
    Returns the circular dispersion of an angle vector.
    
    Parameters
    ----------
    thetas : array-like of shape (n_samples,)
        An angle vector (degree). 
    
    Returns
    -------
    d : float64
        Circular dispersion of the angle vector taken as input.  
    """
    thetas = np.radians(thetas) # degree to radian
    
    R, theta, R2, theta2 = moments(thetas, order=2)
    
    d = (1 - R2)/(2*R**2)
    
    return d


def circular_skew(thetas):
    """
    Returns the circular skewness of an angle vector.
    
    Parameters
    ----------
    thetas : array-like of shape (n_samples,)
        An angle vector (degree). 
    
    Returns
    -------
    s : float64
        Circular skewness of the angle vector taken as input.  
    """
    thetas = np.radians(thetas) # degree to radian
    
    R, theta, R2, theta2 = moments(thetas, order=2)
    
    s = R2*np.sin(theta2 - 2*theta)/(1-R)**(3/2)
    
    return s


def circular_kurtosis(thetas):
    """
    Returns the circular kurtosis of an angle vector.
    
    Parameters
    ----------
    thetas : array-like of shape (n_samples,)
        An angle vector (degree). 
    
    Returns
    -------
    k : float64
        Circular kurtosis of the angle vector taken as input.  
    """
    R, theta, R2, theta2 = moments(thetas, order=2)
    
    k = (R2*np.cos(theta2 - 2*theta) - R**4)/(1-R)**2
    
    return k



def create_sequences_centered(data, sequence_length, t, delta_t_lim=1.05):

    N = len(data)
    #assert N == len(labels)
    assert N == len(t)
    
    # sequence_length = 2*k + 1
    assert sequence_length % 2 == 1
    
    k = (sequence_length - 1)//2
    
    X = []
    #for i in range(k, len(data) - k):
    for i in range(N):
        t_i = t[i]
        X_i = [data[i]]
        for j in range(1, k+1):
            new_index = max(0, min(i-j, N-1))
            if np.abs(t[new_index] - t_i) < j*delta_t_lim:
                X_i.append(data[new_index])
            else:
                X_i.append(X_i[-1])
        X_i.reverse()
        for j in range(1, k+1):
            new_index = max(0, min(i+j, N-1))
            if np.abs(t[new_index] - t_i) < j*delta_t_lim:
                X_i.append(data[new_index])
            else:
                X_i.append(X_i[-1])
        #X.append(data[i-k:i+k+1])  # Input sequence
        X.append(np.array(X_i))
        #y.append(labels[i])  # Label for the sequence
        
    # np.array(X), np.array(y)
    return np.array(X)


def create_dataset(df_gnss, df_position, df_cmc):
    
    # Creation of features
    df = pd.DataFrame()
    
    df['gps_millis'] = np.unique(df_gnss['gps_millis'])
    
    sat_numbers = []
    d_cn0 = {}
    d_elevation = {}
    #d_iono, d_tropo = {}, {}
    #d_sat_bias = {}
    
    for stat in stats:
        d_cn0[stat] = []
        d_elevation[stat] = []
        #d_iono[stat] = []
        #d_tropo[stat] = []
        #d_sat_bias[stat] = []
    
    d_cmc_l1 = []
    d_cmc_e1 = []
    d_cmc_l2 = []
    d_cmc_e5 = []
    d_cmc_utc = []
    
    list_t = np.unique(df_gnss['gps_millis'])
    
    pbar = tqdm(total=len(list_t))
    
    for t in list_t:
    
        pbar.update(1)
    
        # take data with corresponding time with frequency band l1/e1
        sub_data_timestamps = df_gnss[df_gnss['gps_millis']==t] # time
        t_utc = sub_data_timestamps['time'].iloc[0]
        
        indic_l1_e1 = sub_data_timestamps['observation_code'] == '1C' # indic for freq band l1/e1
        sub_data_timestamps = sub_data_timestamps[indic_l1_e1]
        
        unique_sat = np.unique(sub_data_timestamps['gnss_sv_id'])
        sat_numbers.append(len(unique_sat))
    
        cn0_data = sub_data_timestamps['cn0_dbhz'].values
        elevation_data = sub_data_timestamps['el_sv_deg'].values
        #iono_data = sub_data_timestamps['iono_corr'].values
        #tropo_data = sub_data_timestamps['tropo_corr'].values
        #sat_bias_data = sub_data_timestamps['b_sv_m'].values
    
        for stat in stats:

            if len(cn0_data) > 0 and np.sum(np.isnan(cn0_data)) != len(cn0_data):
                val = stats_functions[stat](cn0_data)
            else:
                val = np.nan
            d_cn0[stat].append(val)
            if len(elevation_data) > 0 and np.sum(np.isnan(elevation_data)) != len(elevation_data):
                val = stats_functions[stat](elevation_data)
            else:
                val = np.nan
            d_elevation[stat].append(val)
            #d_iono[stat].append(stats_functions[stat](iono_data))
            #d_tropo[stat].append(stats_functions[stat](tropo_data))
            #d_sat_bias[stat].append(stats_functions[stat](sat_bias_data))
    
        sub_data_timestamps_cmc = df_cmc[df_cmc['gps_millis']==t]
    
        d_cmc_l1.append(np.mean(np.abs(sub_data_timestamps_cmc[sub_data_timestamps_cmc['gnss_id']=='gps']['CMC_a'])))
        d_cmc_e1.append(np.mean(np.abs(sub_data_timestamps_cmc[sub_data_timestamps_cmc['gnss_id']=='galileo']['CMC_a'])))
        d_cmc_l2.append(np.mean(np.abs(sub_data_timestamps_cmc[sub_data_timestamps_cmc['gnss_id']=='gps']['CMC_b'])))
        d_cmc_e5.append(np.mean(np.abs(sub_data_timestamps_cmc[sub_data_timestamps_cmc['gnss_id']=='galileo']['CMC_b'])))
        d_cmc_utc.append(t_utc)
    
    pbar.close()
    
    df['NSV'] = sat_numbers
    
    for stat in stats:
        df['EL '+stat] = np.array(d_elevation[stat])
        df['CN0 '+stat] = np.array(d_cn0[stat])
        #df['Iono_bias '+stat] = np.array(d_iono[stat])
        #df['Tropo_bias '+stat] = np.array(d_tropo[stat])
        #df['SV_bias '+stat] = np.array(d_sat_bias[stat])
    
    # Addition of CMC data
    df['CMC_l1'] = d_cmc_l1
    df['CMC_e1'] = d_cmc_e1
    df['CMC_l2'] = d_cmc_l2
    df['CMC_e5'] = d_cmc_e5

    # Addition of utc time
    df['time_utc'] = d_cmc_utc

    # Specific processing of Nan values for CMC
    #for col in ['CMC_l1', 'CMC_e1', 'CMC_l2', 'CMC_e5']:
    #    if np.isnan(df[col]).sum() == len(df):
    #        df.fillna({col : 0}, inplace=True)
    #    else:
    #        df.fillna({col : np.nanmean(df[col])}, inplace=True)

    # Merging with position estimations
    df_merged = pd.merge(df, df_position, on='gps_millis', how='left')
    #df_merged.dropna(inplace=True)
    
    #df_merged.index = df_merged['time']
    df_merged.index = df_merged['time_utc']
    df_merged.drop('time', axis=1, inplace=True)
    df_merged.drop('time_utc', axis=1, inplace=True)
    df_merged.index = pd.to_datetime(df_merged.index).tz_convert(None)
    df_merged.sort_index(inplace=True)
    
    return df_merged


def find_nearest_line(p, tree, df_class, minimum_index=0):
    
    p_rad = np.deg2rad(p)
    
    distance_i, index_i = tree.query(p_rad.reshape(1,-1))
    index_i = index_i[0,0]
    
    SE = np.array([df_class['longitude_end'].iloc[index_i] - df_class['longitude_start'].iloc[index_i],
                   df_class['latitude_end'].iloc[index_i] - df_class['latitude_start'].iloc[index_i]])
    SE = np.deg2rad(SE)
    SP = np.array([p[0] - df_class['longitude_start'].iloc[index_i],
                   p[1] - df_class['latitude_start'].iloc[index_i]])
    SP = np.deg2rad(SP)
    
    dot_SE_SP = np.dot(SE,SP)
    
    if dot_SE_SP > 0 or index_i == minimum_index:
        index_best = index_i
    else:
        index_best = index_i - 1
        
    return index_best, dot_SE_SP


def fusion_with_gt(df_merged, df_gt, df_class=None):

    try:
        df_gt = df_gt[df_gt['longitude [deg]'] != 0]
        logger.debug("Done with space in column")
    except:
        logger.warning("No space in column")
    
    try:
        df_gt = df_gt[df_gt['longitude'] != 0]
        logger.debug("Done with no space in column")
    except Exception as err:
        logger.warning(err)
            
    df_gt['time'] = pd.to_datetime(df_gt['utc_time'], format='mixed')
    
    tree_time_gt = BallTree(df_gt['time'].values.reshape(-1,1))

    if df_class is not None:
        tree_class = BallTree(np.deg2rad(df_class[['longitude_start', 'latitude_start']].values), metric="haversine")
    
        # Storage of indexes to search in class database
        id_line = np.zeros(len(df_merged), dtype=np.int32)
    
    # Storage of indexes to search in groundtruth
    id_line_loc = np.zeros(len(df_merged), dtype=np.int32)
    
    l_long_lat_points = []
    
    i = 0 # to perform iteration over rows
    for index, row in tqdm(df_merged.iterrows(), total=df_merged.shape[0]):
        
        time_i = np.datetime64(index, 'ns') #+ 18_000_000_000 # Already in UTC Time
    
        _, index_i = tree_time_gt.query(time_i.reshape(-1,1))
        index_in_gt = index_i[0,0]
        id_line_loc[i] = index_in_gt
        
        p = np.array(df_gt.loc[index_in_gt,['longitude','latitude']].values, dtype=np.float64)
        
        l_long_lat_points.append(p)

        if df_class is not None:
        
            index_best, dot_SE_SP = find_nearest_line(p, tree_class, df_class, minimum_index=0)
            
            if dot_SE_SP == 0:
                logger.warning(f"Zero dot with row {index}")
            id_line[i] = index_best
            
        i += 1
    
    l_long_lat_points = np.array(l_long_lat_points)
    
    df_merged['lat_gt'] = l_long_lat_points[:,1]
    df_merged['long_gt'] = l_long_lat_points[:,0]

    if df_class is not None:
    
        df_merged['label'] = df_class['type_env'][id_line].values #le.transform(y_labels)
        id_group = np.zeros(len(df_merged), dtype=np.int32)
        for i in range(1,len(df_merged)):
            if id_line[i] != id_line[i-1] or df_merged['dataset'].iloc[i] != df_merged['dataset'].iloc[i-1]:
                id_group[i] = id_group[i-1] + 1
            else:
                id_group[i] = id_group[i-1]
    
        df_merged['group'] = id_group


def detector_cycle_slip(phi_a, phi_b, thresh_lambda_a_b, N_win):
    
    phi_diff = phi_a - phi_b
    N = len(phi_diff)
    indic = np.zeros(N, dtype=bool)
    
    x_win = np.arange(N_win)
    
    count_wait = 0
    for i in range(N):
        phi_diff_win = phi_diff[i-N_win:i]
        if len(phi_diff_win) == N_win and count_wait == 0:
            p = np.polynomial.Polynomial.fit(x_win, phi_diff_win, deg=2)
            error = np.abs(p(10) - phi_diff[i])
            if error > thresh_lambda_a_b:
                indic[i] = True
                count_wait = N_win
        elif count_wait > 0:
            count_wait -= 1
            
    return indic

def remove_mean(y, indic):
    
    z = np.copy(y)
    
    if np.sum(indic) > 0:
    
        limits = np.where(indic)[0]
    
        i_inf = limits[0]
        z[:i_inf] -= np.mean(y[:i_inf])
        
        if len(limits) > 1:
            for i in range(1, len(limits)):
                i_inf = limits[i-1]
                i_sup = limits[i]
                
                z[i_inf:i_sup] -= np.mean(y[i_inf:i_sup])
        else:
            i_sup = i_inf
            
        z[i_sup:] -= np.mean(y[i_sup:])
    
    else:
        z -= np.mean(y)
    
    return z


def create_multipath_data(df_sv):

    CLIGHT = 299792458.0

    # Dictionary of carrier frequencies for each signal type (in Hz)
    # https://gssc.esa.int/navipedia/index.php?title=GPS_Signal_Plan
    # https://gssc.esa.int/navipedia/index.php?title=Galileo_Signal_Plan
    dict_freq = {'l1' : 1575.42e6,
                 'l2' : 1227.60e6,
                 'l5' : 1176.45e6,
                 'e1' : 1575.42e6,
                 'e5a' : 1176.450e6,
                 'e5b' : 1207.140e6}

    d = dict()
    for id_sat in set(df_sv['gnss_sv_id']):
        d[id_sat] = {}
    
    logger.info("Step CMC 1: read data and store measurements")
    
    for _, row in df_sv.iterrows():
        t = row['gps_millis']
        signal_type = row['signal_type']
        id_sat = row['gnss_sv_id']
        if t not in d[id_sat]:
            d[id_sat][t] = {}
        rho = row['raw_pr_m']
        phi = row['carrier_phase']
        if np.isnan(rho) or np.isnan(phi):
            pass
        else:
            d[id_sat][t][f"{signal_type}-rho"] = rho
            d[id_sat][t][f"{signal_type}-phi"] = phi*CLIGHT/dict_freq[signal_type]
    
    logger.info("Step CMC 2: compute CMC at each time")
    
    for id_sat in d.keys():
        d_sat = d[id_sat]
        for t in d_sat.keys():
            if "l1-rho" in d_sat[t] and "l2-rho" in d_sat[t]:
                # Biased Iono (mean removed latter)
                I_1 = dict_freq['l2']**2/(dict_freq['l1']**2 - dict_freq['l2']**2)*(d_sat[t]["l1-phi"] - d_sat[t]["l2-phi"])
                I_2 = dict_freq['l1']**2/(dict_freq['l1']**2 - dict_freq['l2']**2)*(d_sat[t]["l1-phi"] - d_sat[t]["l2-phi"])
                d_sat[t]['CMC_1'] = d_sat[t]["l1-rho"] - d_sat[t]["l1-phi"] - 2*I_1
                d_sat[t]['CMC_2'] = d_sat[t]["l2-rho"] - d_sat[t]["l2-phi"] - 2*I_2
                
            if "e1-rho" in d_sat[t] and "e5a-rho" in d_sat[t]:
                # Biased Iono (mean removed latter)
                I_1 = dict_freq['e5a']**2/(dict_freq['e1']**2 - dict_freq['e5a']**2)*(d_sat[t]["e1-phi"] - d_sat[t]["e5a-phi"])
                I_5 = dict_freq['e1']**2/(dict_freq['e1']**2 - dict_freq['e5a']**2)*(d_sat[t]["e1-phi"] - d_sat[t]["e5a-phi"])
                d_sat[t]['CMC_1'] = d_sat[t]["e1-rho"] - d_sat[t]["e1-phi"] - 2*I_1
                d_sat[t]['CMC_5'] = d_sat[t]["e5a-rho"] - d_sat[t]["e5a-phi"] - 2*I_5
    
    logger.info("Step CMC 3: filter CMC estimates for each satellite")
    
    df_cmc = []
    
    for id_sat in set(df_sv['gnss_sv_id']):
        
        d_sat = d[id_sat]
        
        if id_sat[0] == 'G':
            X = np.array([[t, d_sat[t]['l1-phi'], d_sat[t]['l2-phi'], d_sat[t]['CMC_1'], d_sat[t]['CMC_2']]  for t in d_sat.keys() if ("l1-rho" in d_sat[t]) and ("l2-rho" in d_sat[t])]) # geometry-free combination
            thresh = CLIGHT/dict_freq['l2'] - CLIGHT/dict_freq['l1']
            gnss_id = 'gps'
        elif id_sat[0] == 'E':
            X = np.array([[t, d_sat[t]['e1-phi'], d_sat[t]['e5a-phi'], d_sat[t]['CMC_1'], d_sat[t]['CMC_5']]  for t in d_sat.keys() if ("e1-rho" in d_sat[t]) and ("e5a-rho" in d_sat[t])]) # geometry-free combination
            thresh = CLIGHT/dict_freq['e5a'] - CLIGHT/dict_freq['e1']
            gnss_id = 'galileo'
        else:
            X = []
            thresh = 0

        if len(X) > 0:
        
            # Warning : following processing need to be made on the whole signal !!!
            y1 = X[:,3]
            y2 = X[:,4]
            
            indic = detector_cycle_slip(X[:,1], X[:,2], thresh, N_win=10)
            zm1 = remove_mean(y1, indic)
            zm2 = remove_mean(y2, indic)
            
            # Now we can separate the signal into different parts.
            t = X[:,0]
            df_cmc_local = pd.DataFrame({'gps_millis' : t, 'CMC_a' : zm1, 'CMC_b' : zm2, 'gnss_id' : gnss_id})
            df_cmc.append(df_cmc_local)

    df_cmc = pd.concat(df_cmc, ignore_index=True)

    return df_cmc


def scale_data(x, mean, std):

    if x.ndim != 2:
        raise ValueError(
            f"Expected 2D input, got input with shape {x.shape}.\n"
            "Reshape your data either using array.reshape(-1, 1) if "
            "your data has a single feature or array.reshape(1, -1) "
            "if it contains a single sample."
            )
    
    return (x - mean)/std


def convert_to_simple_railenium(df):
    '''
    Return simple version of output file with time_rec, count (number of visible satellites) and railenium environments
    '''
    df_alt = pd.DataFrame()

    df_alt['time_rec'] = (
        df['utc_time'].dt.strftime("%Y%m%d-%H%M%S.") +
        (df['utc_time'].dt.microsecond / 1000).round().astype(int).astype(str).str.zfill(3)
        )
    df_alt['count'] = df['count']
    df_alt['environment'] = df['environment']

    # Change 'nan' values
    df_alt.loc[df_alt['environment'] == 'nan', 'environment'] = 'Denied'

    return df_alt


def convert_to_simple_sncf(df):

    df_alt = pd.DataFrame()

    df_alt['time_rec'] = (
        df['utc_time'].dt.strftime("%Y%m%d-%H%M%S.") +
        (df['utc_time'].dt.microsecond / 1000).round().astype(int).astype(str).str.zfill(3)
        )
    df_alt['count'] = df['count']
    df_alt['environment'] = df['environment']

    # Change environment values (hard coded)
    df_alt.loc[df_alt['environment'] == 'tree', 'environment'] = 'Other'
    df_alt.loc[df_alt['environment'] == 'build', 'environment'] = 'Urban denied'
    df_alt.loc[df_alt['environment'] == 'open-sky_rural', 'environment'] = 'Perturbation free'
    df_alt.loc[df_alt['environment'] == 'nan', 'environment'] = 'Denied'

    return df_alt


def get_parser():
    parser = argparse.ArgumentParser(description="Process RINEX files.")
    parser.add_argument("-i", "--input", required=True, help="Path to Groundtruth")
    parser.add_argument("-s", "--svstates", required=True, help="Path to sv_states file")
    parser.add_argument("-w", "--wlssolution", required=True, help="Path to wls_solution file")
    parser.add_argument("-o", "--output", required=True, help="Path to output")
    parser.add_argument("-c", "--config", required=False, help="Path to .ini configuration file")
    parser.add_argument("--outputrail", required=False, help="Path to simple output file with railenium labels")
    parser.add_argument("--outputsncf", required=False, help="Path to simple output file with sncf labels")
    return parser


if __name__ == "__main__":

    circular_stats_functions = {
        'mean' : circular_mean,
        'std' : circular_dispersion,
        'skew' : circular_skew,
        'kurtosis' : circular_kurtosis,
        'min' : np.nanmin,
        'max' : np.nanmax
    }

    stats_functions = {
        'mean' : np.nanmean,
        'std' : np.nanstd,
        'skew' : skew, # nan_policy = 'omit'
        'kurtosis' : kurtosis, # nan_policy = 'omit'
        'min' : np.nanmin,
        'max' : np.nanmax
    }

    args = get_parser().parse_args()
    
    path_gt = args.input
    path_output = args.output
    path_svstates = args.svstates
    path_wlssolution = args.wlssolution
    path_conf_file = args.config
    path_output_rail = args.outputrail
    path_output_sncf = args.outputsncf

    if path_conf_file:
        load_config_from_file(path_conf_file)

    # Model load
    scaler_param = get_scaler_param()
    model_xgb = get_xgb_model()
    model_rnn = get_keras_rnn_model()

    # Data reading
    data_gnsslib = pd.read_csv(path_svstates)
    data_positions = pd.read_csv(path_wlssolution)
    df_gt = pd.read_csv(path_gt)

    # Data features processing
    cmc_data = create_multipath_data(data_gnsslib)
    df_merged = create_dataset(data_gnsslib, data_positions, cmc_data)
    fusion_with_gt(df_merged, df_gt)


    # Dataset preparation for ML
    X = df_merged[feature_names].values
    t = df_merged['gps_millis'].values*1e-3

    # Remove non-float values
    indic = np.isfinite(X).all(axis=1)
    X = X[indic]
    t = t[indic]
    #X_tensor = X_tensor[indic]

    # Stacking of time to create a tensor
    X_tensor = create_sequences_centered(X, sequence_length, t)


    # Scaling
    X = scale_data(X, scaler_param[0,:], scaler_param[1,:])

    for i in range(sequence_length):
        X_tensor[:,i,:] = scale_data(X_tensor[:,i,:], scaler_param[0,:], scaler_param[1,:])

    # Move data to torch tensor
    X_tensor = torch.from_numpy(X_tensor)


    # Prediction

    logger.info("Prediction with ML models")
    pred_xgb = model_xgb.predict_proba(X)
    pred_rnn = model_rnn.predict(X_tensor)


    # Storage
    classes_param = get_classes_param().tolist()
    id_build = classes_param.index('build')
    id_opensky = classes_param.index('open-sky_rural')
    id_tree = classes_param.index('tree')


    df_storage = pd.DataFrame()
    df_storage['utc_time'] = df_merged.index[indic]
    df_storage['lat_gt'] = df_merged['lat_gt'].values[indic]
    df_storage['long_gt'] = df_merged['long_gt'].values[indic]
    df_storage['xgb_build'] = np.array(pred_xgb[:,id_build], dtype=np.float64)
    df_storage['xgb_opensky'] = np.array(pred_xgb[:,id_opensky], dtype=np.float64)
    df_storage['xgb_tree'] = np.array(pred_xgb[:,id_tree], dtype=np.float64)
    df_storage['rnn_build'] = np.array(pred_rnn[:,id_build], dtype=np.float64)
    df_storage['rnn_opensky'] = np.array(pred_rnn[:,id_opensky], dtype=np.float64)
    df_storage['rnn_tree'] = np.array(pred_rnn[:,id_tree], dtype=np.float64)
    df_storage['CMC_l1'] = df_merged['CMC_l1'].values[indic]
    df_storage['CMC_e1'] = df_merged['CMC_e1'].values[indic]
    df_storage['CMC_l2'] = df_merged['CMC_l2'].values[indic]
    df_storage['CMC_e5'] = df_merged['CMC_e5'].values[indic]
    df_storage['count'] = np.array(df_merged['NSV'].values[indic], dtype=int)
    df_storage['blockage'] = 0


    indic_issues = np.logical_not(indic)

    if len(indic_issues) > 0:

        df_blockage = pd.DataFrame()
        df_blockage['utc_time'] = df_merged.index[indic_issues]
        df_blockage['lat_gt'] = df_merged['lat_gt'].values[indic_issues]
        df_blockage['long_gt'] = df_merged['long_gt'].values[indic_issues]
        df_blockage[['xgb_build', 'xgb_opensky', 'xgb_tree']] = np.nan
        df_blockage[['rnn_build', 'rnn_opensky', 'rnn_tree']] = np.nan
        df_blockage['CMC_l1'] = df_merged['CMC_l1'].values[indic_issues]
        df_blockage['CMC_e1'] = df_merged['CMC_e1'].values[indic_issues]
        df_blockage['CMC_l2'] = df_merged['CMC_l2'].values[indic_issues]
        df_blockage['CMC_e5'] = df_merged['CMC_e5'].values[indic_issues]
        df_blockage['count'] = np.array(df_merged['NSV'].values[indic_issues], dtype=int)
        df_blockage['blockage'] = 1

        # Concatenate with previous data
        df_storage = pd.concat([df_storage, df_blockage], ignore_index=True)

    list_env = []
    for val in df_storage[['rnn_build','rnn_opensky','rnn_tree']].values:
        if np.isfinite(val).all():
            list_env.append(classes_param[np.argmax(val)])
        else:
            list_env.append(str(np.nan))
    df_storage['environment'] = list_env

    df_storage.to_csv(path_output, index=False)

    if path_output_rail:
        df_alt_rail = convert_to_simple_railenium(df_storage)
        df_alt_rail.to_csv(path_output_rail, index=False)
    
    if path_output_sncf:
        df_alt_sncf = convert_to_simple_sncf(df_storage)
        df_alt_sncf.to_csv(path_output_sncf, index=False)
