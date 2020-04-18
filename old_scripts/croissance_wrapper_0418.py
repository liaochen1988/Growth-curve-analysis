import numpy as np
import pandas as pd
import copy
import math
import string
from sklearn import metrics
import shutil
import os

from croissance import process_curve
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# note that mu in exponential model is specific growth rate
# while in Gompertz and Logistic model, they are absolute growth rate
def exp_growth_model(x, y1, mu):
    return y1 * np.exp(mu * x)

def exp_growth_model_w_offset(x, y0, y1, mu):
    return y0 + y1 * np.exp(mu * x)

def Zwietering_Gompertz_growth_model(x, y0, A, lag, mu):
    return y0 + A*np.exp(-np.exp(mu*np.exp(1)/A*(lag-x)+1))

def Zwietering_Logistic_growth_model(x, y0, A, lag, mu):
    return y0 + A/(1+np.exp(4*mu/A*(lag-x)+2))

def sgr_Zwietering_Gompertz_growth_model(x, y0, A, lag, mu):
    _exp = np.exp(mu*np.exp(1)/A*(lag-x)+1)
    return A*np.exp(-_exp)*_exp*mu*np.exp(1)/A/(y0+A*np.exp(-_exp))

def sgr_Zwietering_Logistic_growth_model(x, y0, A, lag, mu):
    _exp = np.exp(4*mu/A*(lag-x)+2)
    return 4*mu*_exp/(1+_exp)**2/(y0 + A/(1+_exp))

def get_R2(function_name, popt, x, y):
    if function_name == 'EXP':
        residuals = y-exp_growth_model(x,*popt)
    elif function_name == 'EXP_wOS':
        residuals = y-exp_growth_model_w_offset(x,*popt)
    elif function_name == 'ZG':
        residuals = y-Zwietering_Gompertz_growth_model(x,*popt)
    elif function_name == 'ZL':
        residuals = y-Zwietering_Logistic_growth_model(x,*popt)
    else:
        print('unknown function name.')
        raise
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def fit_data_w_EXP(curr_curve, phase):
    if phase==1 or phase == 2:
        mu_lb = 0
        mu_ub = math.inf
        init_guess = [0.1, 0.1]
    else:
        mu_lb = -math.inf
        mu_ub = 0
        init_guess = [0.1, -0.1]
    best_fits, covar = curve_fit(exp_growth_model,
                                 curr_curve.index,
                                 curr_curve.values,
                                 p0=init_guess,
                                 bounds = ([0,mu_lb], [math.inf,mu_ub]),
                                 maxfev=10000
                                )
    opt_growth_rate = best_fits[1]
    # std_growth_rate = np.sqrt(np.diag(covar))[1]
    R2 = get_R2('EXP', best_fits, curr_curve.index, curr_curve.values)
    return opt_growth_rate, best_fits, R2

def fit_data_w_EXP_wOS(curr_curve, phase):
    if phase==1 or phase == 2:
        mu_lb = 0
        mu_ub = math.inf
        init_guess = [0.1, 0.1, 0.1]
    else:
        mu_lb = -math.inf
        mu_ub = 0
        init_guess = [0.1, 0.1, -0.1]
    best_fits, covar = curve_fit(exp_growth_model_w_offset,
                                 curr_curve.index,
                                 curr_curve.values,
                                 p0=init_guess,
                                 bounds = ([0,0,mu_lb], [math.inf,math.inf,mu_ub]),
                                 maxfev=10000
                                )
    opt_growth_rate = best_fits[2]
    # std_growth_rate = np.sqrt(np.diag(covar))[2]
    R2 = get_R2('EXP_wOS', best_fits, curr_curve.index, curr_curve.values)
    return opt_growth_rate, best_fits, R2

def fit_data_w_ZG(curr_curve, phase):
    if phase==1 or phase == 2:
        mu_lb = 0
        mu_ub = math.inf
        init_guess = [0.1, 1.0, np.median(curr_curve.index), 0.1]
    else:
        mu_lb = -math.inf
        mu_ub = 0
        init_guess = [0.1, 1.0, np.median(curr_curve.index), -0.1]
    best_fits, covar = curve_fit(Zwietering_Gompertz_growth_model,
                                 curr_curve.index,
                                 curr_curve.values,
                                 p0=init_guess,
                                 bounds = ([0,0,curr_curve.index[0],mu_lb], [math.inf,math.inf,curr_curve.index[-1],mu_ub]),
                                 maxfev=10000
                                )
    opt_growth_rate = max(sgr_Zwietering_Gompertz_growth_model(curr_curve.index, *best_fits))
    R2 = get_R2('ZG', best_fits, curr_curve.index, curr_curve.values)
    return opt_growth_rate, best_fits, R2

def fit_data_w_ZL(curr_curve, phase):
    if phase==1 or phase == 2:
        mu_lb = 0
        mu_ub = math.inf
        init_guess = [0.1, 1.0, np.median(curr_curve.index), 0.1]
    else:
        mu_lb = -math.inf
        mu_ub = 0
        init_guess = [0.1, 1.0, np.median(curr_curve.index), -0.1]
    best_fits, covar = curve_fit(Zwietering_Logistic_growth_model,
                                 curr_curve.index,
                                 curr_curve.values,
                                 p0=init_guess,
                                 bounds = ([0,0,curr_curve.index[0],mu_lb], [math.inf,math.inf,curr_curve.index[-1],mu_ub]),
                                 maxfev=10000
                                )
    opt_growth_rate = max(sgr_Zwietering_Logistic_growth_model(curr_curve.index, *best_fits))
    R2 = get_R2('ZL', best_fits, curr_curve.index, curr_curve.values)
    return opt_growth_rate, best_fits, R2

def growth_curve_analysis_chenL(infile,
                                blank=None,
                                which_curves=[x+str(y) for y in range(1,13) for x in string.ascii_uppercase[:8]],
                                OD_cutoff=0.001,
                                verbose=0,
                                is_plot=False,
                                plot_dim=(8,12),
                                fitting_method_in_plot='Zwietering_Logistic',
                                is_write=False,
                                xlim=None,
                                ylim=None,
                                well_name_dict={}
                               ):

    '''
    infile: output file of 96-well plate reader
    blank: row of column used as blank control wells
    which_curves: which plate wells will be analyzed
    OD_cutoff: OD smaller than this value will be set to this value
    verbose: how much intermediate processing results are printed to screen
    is_plot: whether to plot curve fitting
    plot_dim: number of rows and columns in a plot. by default, it is 8 x 12
    fitting_method_in_plot: fitted curve using the specified fitting method will be shown in the plot
                            choose between "exp", "exp_w_offset", "ZG", "ZL"
    is_write: whether to write extracted growth curve parameters into file
    xlim: x-axis range if is_plot is true
    ylim: y-axis range if is_plot is true
    well_name_dict: user-provided name of each of 96 wells
    '''
    #--------------
    # preprocessing
    #--------------
    
    # read data 
    # use time as index (index_col=1)
    if infile.split('.')[-1] == 'xlsx':
        df = pd.read_excel(infile, index_col=1)
    elif infile.split('.')[-1] == 'csv':
        df = pd.read_csv(infile, index_col=1)
    else:
        print('file format not recognized. Support xlsx and csv only.')
        raise

    # check if index name is time
    if df.index.name != 'Time [s]':
        print("dataframe index must be Time [s]. current value is %s."%(df.index.name))
        raise
    else:
        # convert time unit to hour
        df.index = df.index/3600
        df.index.names = ['Time (h)']

    # remove unnecessary columns (cycle and temperature)
    df = df.drop(['Cycle Nr.', 'Temp. [°C]'], axis=1)

    # subtract baseline data (controls)
    if blank is not None:
        df_res = copy.deepcopy(df)

        refrows = [x for x in blank if x.isalpha()] # A-H
        for letter in refrows:
            for idx, colname in enumerate(df.columns):
                digits = ''.join(i for i in colname if i.isdigit()) # digits in the column name, e.g., returns '8' for A8
                colname2subtract = df.columns.values.tolist().index(letter + digits)
                #print(colname, letter + digits,colname2subtract)
                df_res.iloc[:,idx] = df_res.iloc[:,idx] - df.iloc[:,colname2subtract]/len(blank)

        refcols = [x for x in blank if x.isdigit()] # 1-12
        for digits in refcols:
            for idx, colname in enumerate(df.columns):
                letter = ''.join(i for i in colname if i.isalpha())
                colname2subtract = df.columns.values.tolist().index(letter + digits)
                #print(colname, letter + digits,colname2subtract)
                df_res.iloc[:,idx] = df_res.iloc[:,idx] - df.iloc[:,colname2subtract]/len(blank)

        df = copy.deepcopy(df_res)

    # plot fitting
    if is_plot:
        fig, axes = plt.subplots(nrows=plot_dim[0], ncols=plot_dim[1], squeeze=False, figsize=(16, 10))
        
#     # write rank threshold into file
#     shutil.copy('./croissance/estimation/defaults.py','./croissance/estimation/defaults.py.bak')
#     with open('./croissance/estimation/defaults.py') as f:
#         content = f.readlines()
#         for index, line in enumerate(content):
#             if 'PHASE_RANK_EXCLUDE_BELOW' in line:
#                 index_of_rank_threshold = index
#                 break
#         content[index_of_rank_threshold] = 'PHASE_RANK_EXCLUDE_BELOW = %d\n'%(rank_threshold)
#     with open("./croissance/estimation/defaults.py", "w") as output:
#         output.writelines('%s' % s for s in content)
    
    #------------------------------------
    # growth curve analysis starts here
    #------------------------------------
    line2append = []
    well_index = -1
    alphabetic_letters = string.ascii_uppercase[:8]
    for colname in df.columns:
        if colname not in which_curves:
            if verbose==1:
                print(colname, ' skipped.')
            continue
        else:
            well_index += 1
            if colname in well_name_dict:
                colname_readable = well_name_dict[colname]
            else:
                colname_readable = None

        # plot data
        if is_plot:
            axes_row_index = int(well_index % plot_dim[0])
            axes_col_index = int(np.floor(well_index/plot_dim[0]))
            # print(axes_row_index, axes_col_index)

            # original data
            axes[axes_row_index, axes_col_index].plot(df.index,
                                      df[colname].values,
                                      color='black',
                                      marker='.',
                                      markersize=2,
                                      linestyle='None')

            # set xlim
            if xlim is not None:
                axes[axes_row_index, axes_col_index].set_xlim(xlim)
            else:
                axes[axes_row_index, axes_col_index].set_xlim([0,50])

            # set ylim
            if ylim is not None:
                axes[axes_row_index, axes_col_index].set_ylim(ylim)
            else:
                axes[axes_row_index, axes_col_index].set_ylim([0.005, 2])

            axes[axes_row_index, axes_col_index].set_yscale('log')
            axes[axes_row_index, axes_col_index].minorticks_off()

            # set xlabel and ticks
            if axes_row_index != plot_dim[0]-1:
                axes[axes_row_index, axes_col_index].set_xlabel('')
                axes[axes_row_index, axes_col_index].set_xticklabels([])
            else:
                axes[axes_row_index, axes_col_index].set_xlabel('Time (h)')
                axes[axes_row_index, axes_col_index].set_xticks([0,10,20,30,40,50])

            # set ylabel and ticks
            if axes_col_index != 0:
                axes[axes_row_index, axes_col_index].set_ylabel('')
                axes[axes_row_index, axes_col_index].set_yticklabels([])
            else:
                axes[axes_row_index, axes_col_index].set_ylabel('OD')
                axes[axes_row_index, axes_col_index].set_yticks([0.01,0.1,1])

            # set title
            if colname_readable is None:
                axes[axes_row_index, axes_col_index].set_title(colname)
            else:
                axes[axes_row_index, axes_col_index].set_title(colname_readable)

        # OD below than OD_cutoff will be imputed by backfill first then forward fill
        curr_df = df[colname]
        curr_df[curr_df < OD_cutoff] = np.nan
        curr_df = curr_df.fillna(method='backfill')
        if (curr_df.isnull().any()):
            curr_df = curr_df.fillna(method='ffill')
            if (curr_df.isnull().all()):
                if verbose==1:
                    print('%s: no data above OD cutoff.'%(colname))
                continue
        # print(curr_df)
        
        # find three growth phases
        # phase 1: positive first-order derivative and positive second-order derivative
        # phase 2: positive first-order derivativev and negative second-order derivative
        # phase 3: negative first-order derivative
        phase = [1,2,3]
        first_derivative_sign = [1, 1, -1]
        second_derivative_sign = [1, -1, 0]  
        phase_color = ['skyblue', 'limegreen', 'violet']
        for p, fds, sds, pcolor in zip(phase, first_derivative_sign, second_derivative_sign, phase_color):            
            # call croissance
            results = process_curve(curr_df, 
                                    constrain_n0=True, 
                                    n0=0., 
                                    first_derivative_sign=fds,
                                    second_derivative_sign=sds,
                                   )
       
            # order subphases in time
            start_time = [subphase.start for subphase in results.growth_phases]
            if len(start_time) != 0:
                subphase_order = np.argsort(start_time)
            else:
                if verbose==1:
                    print('%s: phase %d not found.'%(colname, p))
                continue

            # loop over growth phases          
            for subphase_index_new, subphase_index_old in enumerate(subphase_order):
                curr_subphase = results.growth_phases[subphase_index_old]
                curr_curve = results.series[curr_subphase.start:curr_subphase.end]

                # see if slope sign is expected
                if (p==1 or p==2) and curr_subphase.slope < 0:
                    continue
                if p==3 and curr_subphase.slope > 0:
                    continue
                
                # fit phase curve using different models
                opt_growth_rate_EXP, best_fits_EXP, R2_EXP = fit_data_w_EXP(curr_curve, p)
                opt_growth_rate_EXP_wOS, best_fits_EXP_wOS, R2_EXP_wOS = fit_data_w_EXP_wOS(curr_curve, p)
                opt_growth_rate_ZG, best_fits_ZG, R2_ZG = fit_data_w_ZG(curr_curve, p)
                opt_growth_rate_ZL, best_fits_ZL, R2_ZL = fit_data_w_ZL(curr_curve, p)
                      
                # add data to output
                line2append.append({'id':colname,
                                    'name':colname_readable,
                                    'phase':p,
                                    'subphase':alphabetic_letters[subphase_index_new],
                                    'start_time':curr_subphase.start,
                                    'end_time':curr_subphase.end,
                                    'start_OD':curr_curve.values[0],
                                    'end_OD':curr_curve.values[-1],
                                    'area':metrics.auc(curr_curve.index,curr_curve.values),
                                    'growth_rate_EXP':opt_growth_rate_EXP,
                                    'R2_EXP':R2_EXP,
                                    'growth_rate_EXP_wOS':opt_growth_rate_EXP_wOS,
                                    'R2_EXP_wOS':R2_EXP_wOS,
                                    'growth_rate_ZG':opt_growth_rate_ZG,
                                    'R2_ZG':R2_ZG,
                                    'growth_rate_ZL':opt_growth_rate_ZL,
                                    'R2_ZL':R2_ZL
                                   })
                if verbose==1:
                    print('%s, phase %s, [%2.2f, %2.2f], growth rate = [%2.4f, %2.4f, %2.4f, %2.4f], R2=[%2.2f, %2.2f, %2.2f, %2.2f]' %
                          (colname, str(p)+alphabetic_letters[subphase_index_new], curr_subphase.start, curr_subphase.end,
                           opt_growth_rate_EXP, opt_growth_rate_EXP_wOS, opt_growth_rate_ZG, opt_growth_rate_ZL,
                           R2_EXP, R2_EXP_wOS, R2_ZG, R2_ZL))

                # plot
                if is_plot:
                    # phase contour
                    axes[axes_row_index, axes_col_index].plot(
                        curr_curve.index,
                        curr_curve.values,
                        marker=None,
                        linewidth=5,
                        color=pcolor,
                        solid_capstyle='butt',
                        alpha=1.0)

                    # model fitting
                    fitted_curve = None
                    if fitting_method_in_plot == 'EXP':
                        fitted_curve = exp_growth_model(results.series.index, *best_fits_EXP)
                    elif fitting_method_in_plot == 'EXP_wOS':
                        fitted_curve = exp_growth_model_w_offset(results.series.index, *best_fits_EXP_wOS)
                    elif fitting_method_in_plot == 'ZG':
                        fitted_curve = Zwietering_Gompertz_growth_model(results.series.index, *best_fits_ZG)
                    elif fitting_method_in_plot == 'ZL':
                        fitted_curve = Zwietering_Logistic_growth_model(results.series.index, *best_fits_ZL)
                    else:
                        print('unknown fitting method.')
                        raise
                    if fitted_curve is not None:
                        axes[axes_row_index, axes_col_index].plot(
                            results.series.index,
                            fitted_curve,
                            color=pcolor,
                            linewidth=1.5,
                            linestyle='--')

    if is_plot:
        plt.show()
        plt.tight_layout()

    # write to file
    df2write = pd.DataFrame(line2append)
    if is_write:
        df2write = pd.DataFrame(line2append)
        df2write.to_excel('output/'+infile.split('.xlsx')[0].split('/')[-1]+'__growth_curve_analysis.xlsx')
        
#     # copy back
#     shutil.copy('./croissance/estimation/defaults.py.bak','./croissance/estimation/defaults.py')
#     os.remove("./croissance/estimation/defaults.py.bak")
    
    return df2write
