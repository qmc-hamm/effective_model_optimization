import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter


def plot_spectrumloss(df:pd.DataFrame, w0:float, temp:float):

    dir_plots = "plots"
    if not os.path.exists(dir_plots):
        os.makedirs(dir_plots)

    plt.figure()
    axes = plt.axes(projection = '3d')

    x = df['RDMD doccp (eV)']
    y = df['RDMD t_1 (eV)']
    z = df['Spectrum RMSE Test natoms4_casscf (eV)']
    color = df['RDMD trace (eV)'] #loss

    cax = axes.scatter(x, y, z, c=color)

    cb = plt.colorbar(cax, label='E0')
    cb.formatter.set_useOffset(False)
    cb.update_ticks()
    axes.set_xlabel("U (eV)")
    axes.set_ylabel("t (eV)")
    axes.set_xlim(0, 12)
    axes.set_ylim(-4, 4)
    axes.set_zlabel("Spectrum RMSE (eV)")
    plt.ticklabel_format(useOffset=False)
    plt.savefig(f'{dir_plots}/w0{w0}_temp{temp}_spectrumloss.png')

    plt.close()

    return


def plot_loss(df:pd.DataFrame, w0:float, temp:float):

    dir_plots = "plots"
    if not os.path.exists(dir_plots):
        os.makedirs(dir_plots)

    plt.figure()
    axes = plt.axes(projection = '3d')

    x = df['RDMD doccp (eV)']
    y = df['RDMD t_1 (eV)']
    z = df['loss']
    color = df['RDMD trace (eV)'] #loss

    cax = axes.scatter(x, y, z, c=color)

    cb = plt.colorbar(cax, label='E0')
    cb.formatter.set_useOffset(False)
    axes.set_xlabel("U (eV)")
    axes.set_ylabel("t (eV)")
    axes.set_xlim(0, 12)
    axes.set_ylim(-4, 4)
    axes.set_zlabel("loss")
    plt.ticklabel_format(useOffset=False)
    plt.savefig(f'{dir_plots}/w0{w0}_temp{temp}_loss.png')

    plt.close()

    return


def plot_compare_w0(w0s, temp):
    dir_plots = "plots"
    if not os.path.exists(dir_plots):
        os.makedirs(dir_plots)
    
    plt.figure()
    axes = plt.axes(projection = '3d')

    dir = "runs"
    for w0 in w0s:
        try:
            df = pd.read_csv(f"{dir}/w0{w0}_temp{temp}/Processed_CV_data.csv")
        except:
            print(f"No Folder : {dir}/w0{w0}_temp{temp}/")

        x = df['RDMD doccp (eV)']
        y = df['RDMD t_1 (eV)']
        z = df['Spectrum RMSE Test natoms4_casscf (eV)']
        color = np.ones(len(x))*w0

        cax = axes.scatter(x, y, z, c=color, vmin=min(w0s), vmax=max(w0s))

    cb = plt.colorbar(cax, label='w0')
    cb.formatter.set_useOffset(False)
    axes.set_xlabel("U (eV)")
    axes.set_ylabel("t (eV)")
    axes.set_xlim(0, 12)
    axes.set_ylim(-4, 4)
    axes.set_zlabel("Spectrum RMSE (eV)")
    plt.ticklabel_format(useOffset=False)
    
    plt.savefig(f'{dir_plots}/compare_w0s_temp{temp}_spectrumloss.png')

    plt.close()
        

if __name__=="__main__":

    dir = "runs"
    w0s = [1.0, 0.9, 0.8, 0.7]
    temps = [0.0, 2.5, 5.0, 10.0, 15.0]

    for w0 in w0s:
        for temp in temps:
            try:
                df = pd.read_csv(f"{dir}/w0{w0}_temp{temp}/Processed_CV_data.csv")
            except:
                print(f"No Folder : {dir}/w0{w0}_temp{temp}/")

            plot_spectrumloss(df, w0, temp)
            plot_loss(df, w0, temp)

    
    for temp in temps:
        plot_compare_w0(w0s, temp)
