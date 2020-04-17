import numpy as np
import pandas as pd
import copy
import math
import string

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

def get_R2(function_name, popt, x, y, R2_logscale):
    if function_name == 'exp':
        if R2_logscale==False:
            residuals = y-exp_growth_model(x,*popt)
        else:
            residuals = np.log10(y)-np.log10(exp_growth_model(x,*popt))
    elif function_name == 'exp_w_offset':
        if R2_logscale==False:
            residuals = y-exp_growth_model_w_offset(x,*popt)
        else:
            residuals = np.log10(y)-np.log10(exp_growth_model_w_offset(x,*popt))
    elif function_name == 'ZG':
        if R2_logscale==False:
            residuals = y-Zwietering_Gompertz_growth_model(x,*popt)
        else:
            residuals = np.log10(y)-np.log10(Zwietering_Gompertz_growth_model(x,*popt))
    elif function_name == 'ZL':
        if R2_logscale==False:
            residuals = y-Zwietering_Logistic_growth_model(x,*popt)
        else:
            residuals = np.log10(y)-np.log10(Zwietering_Logistic_growth_model(x,*popt))
    else:
        print('unknown function name.')
        raise
    ss_res = np.sum(residuals ** 2)
    if R2_logscale==False:
        ss_tot = np.sum((y - np.mean(y)) ** 2)
    else:
        ss_tot = np.sum((np.log10(y) - np.mean(np.log10(y))) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def growth_curve_analysis_chenL(infile,
                                blank=None,
                                which_curves=[x+str(y) for y in range(1,13) for x in string.ascii_uppercase[:8]],
                                OD_cutoff=0.001,
                                verbose=0,
                                is_plot=False,
                                plot_dim=(8,12),
                                fitting_method_in_plot='Zwietering_Logistic',
                                R2_logscale=False,
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
    R2_logscale: whether to calculate R2 in log10 scale
    is_write: whether to write extracted growth curve parameters into file
    xlim: x-axis range if is_plot is true
    ylim: y-axis range if is_plot is true
    well_name_dict: user-provided name of each of 96 wells
    '''

    # read data
    # use time as index
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

    # growth curve analysis
    data_2_save = []
    well_index = -1
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
                colname_readable = colname

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
                axes[axes_row_index, axes_col_index].set_ylim([0.0005, 2])

            axes[axes_row_index, axes_col_index].set_yscale('log')
            axes[axes_row_index, axes_col_index].minorticks_off()

            # set xlabel and ticks
            if axes_row_index != plot_dim[0]-1:
                axes[axes_row_index, axes_col_index].set_xlabel('')
                axes[axes_row_index, axes_col_index].set_xticklabels([])
            else:
                axes[axes_row_index, axes_col_index].set_xlabel('Time (h)')
                axes[axes_row_index, axes_col_index].set_xticks([0, 25, 50])

            # set ylabel and ticks
            if axes_col_index != 0:
                axes[axes_row_index, axes_col_index].set_ylabel('')
                axes[axes_row_index, axes_col_index].set_yticklabels([])
            else:
                axes[axes_row_index, axes_col_index].set_ylabel('OD')
                axes[axes_row_index, axes_col_index].set_yticks([0.001,0.01,0.1,1])

            # set title
            axes[axes_row_index, axes_col_index].set_title(colname_readable)

        # OD below than a threshold will be imputed by values above the threshold
        df2do = df[colname]
        df2do[df2do < OD_cutoff] = np.nan
        df2do = df2do.fillna(method='backfill')
        if (df2do.isnull().any()):
            df2do = df2do.fillna(method='ffill')
            if (df2do.isnull().all()):
                if verbose==1:
                    print('%s: no data above OD cutoff.'%(colname))
                data_2_save.append({'curve':colname_readable,
                                    'well':colname,
                                    'phase':np.nan,
                                    'start_time':np.nan,
                                    'end_time':np.nan,
                                    'growth_rate_EXP':np.nan,
                                    'growth_ratt_std_exp':np.nan,
                                    'R2_EXP':np.nan,
                                    'std_EXP_wOS':np.nan,
                                    'growth_rate_EXP_wOS':np.nan,
                                    'R2_EXP_wOS':np.nan,
                                    'growth_rate_ZG':np.nan,
                                    'R2_ZG':np.nan,
                                    'growth_rate_ZL':np.nan,
                                    'R2_ZL':np.nan
                                    })
                continue

        # run Croissance
        try:
            results = process_curve(df2do, constrain_n0=True, n0=0.)
        except Exception as error:
            print('error in croissance: %s'%(str(error)))
            raise

        # order phases in time
        start_time = []
        for phase in results.growth_phases:
            start_time.append(phase.start)
        if start_time:
            phase_order = np.argsort(start_time)
        else:
            if verbose==1:
                print('%s: no phase found.'%(colname))
            data_2_save.append({'curve':colname_readable,
                                'well':colname,
                                'phase':np.nan,
                                'start_time':np.nan,
                                'end_time':np.nan,
                                'growth_rate_EXP':np.nan,
                                'growth_ratt_std_exp':np.nan,
                                'R2_EXP':np.nan,
                                'std_EXP_wOS':np.nan,
                                'growth_rate_EXP_wOS':np.nan,
                                'R2_EXP_wOS':np.nan,
                                'growth_rate_ZG':np.nan,
                                'R2_ZG':np.nan,
                                'growth_rate_ZL':np.nan,
                                'R2_ZL':np.nan
                               })
            continue

        # loop over growth phases
        phase_color = ['skyblue', 'limegreen', 'violet', 'gold']
        for phase_index_new, phase_index_old in enumerate(phase_order):
            curr_phase = results.growth_phases[phase_index_old]
            curr_curve = results.series[curr_phase.start:curr_phase.end]

            '''
            # fit phase curve using different models
            '''

            # exponential
            init_guess = [1/np.exp(curr_phase.intercept * curr_phase.slope), curr_phase.slope] # initial guess for [y1, mu]
            best_fits_exp, covar_exp = curve_fit(exp_growth_model,
                                                 curr_curve.index,
                                                 curr_curve.values,
                                                 p0=init_guess,
                                                 bounds = ([0,0], [math.inf,math.inf]),
                                                 maxfev=10000)
            curr_gr_exp = best_fits_exp[1]
            curr_gr_std_exp = np.sqrt(np.diag(covar_exp))[1]
            R2_EXP = get_R2('exp', best_fits_exp, curr_curve.index, curr_curve.values, R2_logscale)

            # exponential with offset
            init_guess = [0, 1/np.exp(curr_phase.intercept * curr_phase.slope), curr_phase.slope] # initial guess for [y0, y1, mu]
            best_fits_exp_w_offset, covar_exp_w_offset = curve_fit(exp_growth_model_w_offset,
                                                                   curr_curve.index,
                                                                   curr_curve.values,
                                                                   p0=init_guess,
                                                                   bounds = ([0,0,0], [math.inf,math.inf,math.inf]),
                                                                   maxfev=10000)
            curr_gr_exp_w_offset = best_fits_exp_w_offset[2]
            curr_gr_std_exp_w_offset = np.sqrt(np.diag(covar_exp_w_offset))[2]
            R2_EXP_wOS = get_R2('exp_w_offset', best_fits_exp_w_offset, curr_curve.index, curr_curve.values, R2_logscale)

            # Zwiesttering_Gompertz
            init_guess = [curr_curve.values[0], max(curr_curve.values), curr_phase.start, curr_phase.slope] # initial guess for [y0, A, lag, mu]
            best_fits_ZG, covar_ZG = curve_fit(Zwietering_Gompertz_growth_model,
                                               curr_curve.index,
                                               curr_curve.values,
                                               p0=init_guess,
                                               bounds = ([0,0,curr_phase.start,0], [math.inf,math.inf,curr_phase.end,math.inf]),
                                               maxfev=10000)
            curr_gr_ZG = max(sgr_Zwietering_Gompertz_growth_model(curr_curve.index, *best_fits_ZG))
            R2_ZG = get_R2('ZG', best_fits_ZG, curr_curve.index, curr_curve.values, R2_logscale)

            # Zwiesttering_Logistic
            init_guess = [curr_curve.values[0], max(curr_curve.values), curr_phase.start, curr_phase.slope] # initial guess for [y0, A, lag, mu]
            best_fits_ZL, covar_ZL = curve_fit(Zwietering_Logistic_growth_model,
                                               curr_curve.index,
                                               curr_curve.values,
                                               p0=init_guess,
                                               bounds = ([0,0,curr_phase.start,0], [math.inf,math.inf,curr_phase.end,math.inf]),
                                               maxfev=10000)
            curr_gr_ZL = max(sgr_Zwietering_Logistic_growth_model(curr_curve.index, *best_fits_ZL))
            R2_ZL = get_R2('ZL', best_fits_ZL, curr_curve.index, curr_curve.values, R2_logscale)

            # save data
            data_2_save.append({'curve':colname_readable,
                                'well':colname,
                                'phase':phase_index_new+1,
                                'start_time':curr_phase.start,
                                'end_time':curr_phase.end,
                                'growth_rate_EXP':curr_gr_exp,
                                'std_EXP':curr_gr_std_exp,
                                'R2_EXP':R2_EXP,
                                'growth_rate_EXP_wOS':curr_gr_exp_w_offset,
                                'std_EXP_wOS':curr_gr_std_exp_w_offset,
                                'R2_EXP_wOS':R2_EXP_wOS,
                                'growth_rate_ZG':curr_gr_ZG,
                                'R2_ZG':R2_ZG,
                                'growth_rate_ZL':curr_gr_ZL,
                                'R2_ZL':R2_ZL
                               })
            if verbose==1:
                print('%s (%d phases), phase %d ([%2.2f, %2.2f]), growth rate = [%2.4f, %2.4f, %2.4f, %2.4f], std = [%2.4f, %2.4f], R2=[%2.2f, %2.2f, %2.2f, %2.2f]' %
                      (colname, len(results.growth_phases), phase_index_new+1, curr_phase.start, curr_phase.end,
                       curr_gr_exp, curr_gr_exp_w_offset, curr_gr_ZG, curr_gr_ZL,
                       curr_gr_std_exp, curr_gr_std_exp_w_offset,
                       R2_EXP, R2_EXP_wOS, R2_ZG, R2_ZL))

            # plot
            if is_plot:
                # phase contour
                axes[axes_row_index, axes_col_index].plot(
                    curr_curve.index,
                    curr_curve.values,
                    marker=None,
                    linewidth=5,
                    color=phase_color[phase_index_new],
                    solid_capstyle='butt',
                    alpha=1.0)

                # model fitting
                fitted_curves = []
                if 'exp' in fitting_method_in_plot or 'all' in fitting_method_in_plot:
                    fitted_curves.append(exp_growth_model(results.series.index, *best_fits_exp))
                if 'exp_w_offset' in fitting_method_in_plot or 'all' in fitting_method_in_plot:
                    fitted_curves.append(exp_growth_model_w_offset(results.series.index, *best_fits_exp_w_offset))
                if 'ZG' in fitting_method_in_plot or 'all' in fitting_method_in_plot:
                    fitted_curves.append(Zwietering_Gompertz_growth_model(results.series.index, *best_fits_ZG))
                if 'ZL' in fitting_method_in_plot or 'all' in fitting_method_in_plot:
                    fitted_curves.append(Zwietering_Logistic_growth_model(results.series.index, *best_fits_ZL))
                if len(fitted_curves)>0:
                    colors = ['olive','mediumslateblue','cyan','pink']
                    for color_index, fc in enumerate(fitted_curves):
                        axes[axes_row_index, axes_col_index].plot(
                            results.series.index,
                            fc,
                            color=colors[color_index],
                            linewidth=1.5,
                            linestyle='--')

    if is_plot:
        plt.show()
        plt.tight_layout()

    # write to file
    df2write = pd.DataFrame(data_2_save)
    if is_write:
        df2write = pd.DataFrame(data_2_save)
        df2write.to_excel(infile.split('.xlsx')[0].split('/')[-1]+'__growth_phase_analysis.xlsx')

    return df2write
