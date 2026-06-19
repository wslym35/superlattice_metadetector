#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:33:33 2026

@author: quadrupole
"""
import sys, os 

sys.path.append("/opt/lumerical/v251/api/python") # for laptop
sys.path.append("/opt/lumerical/v241/api/python") # for desktop 
sys.path.append(os.path.dirname(__file__)) 
import lumapi

import numpy as np
#import h5py
import matplotlib.pyplot as plt
import gc
import time 
import traceback 

#from dOpt import min_mesa_width 
min_mesa_width = 50e-9 
#minimum_ribbon_thickness = 0.050e-6 
c = 3e8 # Speed of light, m/s 
epsilon0 = 8.854187817e-12 # F/m 

def setup(params, sim): 
    
    # Clear anything left over from previous runs 
    sim.switchtolayout() 
    sim.selectall()
    sim.delete() 
    
    # Build the metasurface 
    # z=0 is the bottom of the substrate, so all z coordinates are > 0 
    sim.addrcwa() # Add rcwa region 
    sim.set("name", "RCWA")
    
    sim.addrcwafieldmonitor() 
    sim.set("name", "monitor") 
    
    configuration = [
        ("RCWA", ( 
            ("x min", 0), ("x max", params['period'][0]),
            ("y min", 0), ("y max", params['period'][1]),
            ("z min", 0), ("z max", sum(params['layer_thicknesses'][:])),
            # set excitation wavelength here (M-point broadband is an option, but memory-intensive) 
            ("frequency points", len(params['wavelength_range'])), ("custom frequency samples", c/params['wavelength_range']),
            # Set RCWA interfaces 
            ("interface absolute positions", np.array([sum(params['layer_thicknesses'][0:i]) for i in range(1, params['layer_count'])])), 
            # Set the number of k vectors to use in the RCWA solver 
            ("max number k vectors", params['Fourier_N']), 
            # Record index to find n_sapp 
            #("report index", True),
            )), 
        ("monitor", (
            ("x min", 0), ("x max", params['period'][0]), ("x number points", params['xy_mesh']),
            ("y min", 0), ("y max", params['period'][1]), ("y number points", params['xy_mesh']),
            ("z min", sum(params['layer_thicknesses'][0:1])), ("z max", sum(params['layer_thicknesses'][0:2])), ("z number points", params['z_mesh']),
            # Only need to record Ex, Ey, and Ez 
            ("output Ex", True), ("output Ey", True),("output Ez", True), 
            ("output Hx", False), ("output Hy", False), ("output Hz", False),
            ))
        ]
    
    for i in range(params['layer_count']):
        
        if not params['layer_is_etched'][i]: 
            sim.addrect()
            sim.set("name", params['layer_names'][i]) 
            configuration.append(
                (params['layer_names'][i], ( 
                    ("x min", 0), ("x max", params['period'][0]),
                    ("y min", 0), ("y max", params['period'][1]),
                    ("z min", sum(params['layer_thicknesses'][0:i])), ("z max", sum(params['layer_thicknesses'][0:i+1])),
                    ("material", params['layer_materials'][i])
                    ))
                )
            
        else: # If the layer is etched 
            for r in range(params['ribbon_count']):
                sim.addrect()
                sim.set("name", f"{params['layer_names'][i]}, ribbon {r}")
                configuration.append(
                    (f"{params['layer_names'][i]}, ribbon {r}", (
                        ("x", params['ribbon_centers'][r]), ("x span", params['ribbon_widths'][r]),
                        ("y min", 0), ("y max", params['period'][1]),
                        ("z min", sum(params['layer_thicknesses'][0:i])), ("z max", sum(params['layer_thicknesses'][0:i+1])),
                        ("material", params['layer_materials'][i])
                        ))
                    )
                for n in range(params['notch_count']):
                    # notches are carved into both sides of the ribbon, to a depth such that the remaining ribbon width is "min_mesa_width"
                    # checked 2026-03-24 
                    sim.addrect()
                    sim.set("name", f"{params['layer_names'][i]}, ribbon {r}, notch {n}, negative side")
                    configuration.append(
                        (f"{params['layer_names'][i]}, ribbon {r}, notch {n}, negative side", (
                            ("x min", params['ribbon_centers'][r] - params['ribbon_widths'][r]/2), ("x max", params['ribbon_centers'][r] - min_mesa_width/2),
                            ("y", params['notch_centers'][n]), ("y span", params['notch_widths'][n]),
                            ("z min", sum(params['layer_thicknesses'][0:i])), ("z max", sum(params['layer_thicknesses'][0:i+1])),
                            ("material", "etch")
                            ))
                        )
                    
                    sim.addrect()
                    sim.set("name", f"{params['layer_names'][i]}, ribbon {r}, notch {n}, positive side")
                    configuration.append(
                        (f"{params['layer_names'][i]}, ribbon {r}, notch {n}, positive side", (
                            ("x min", params['ribbon_centers'][r] + min_mesa_width/2), ("x max", params['ribbon_centers'][r] + params['ribbon_widths'][r]/2),
                            ("y", params['notch_centers'][n]), ("y span", params['notch_widths'][n]),
                            ("z min", sum(params['layer_thicknesses'][0:i])), ("z max", sum(params['layer_thicknesses'][0:i+1])),
                            ("material", "etch")
                            ))
                        )
    
    for obj, parameters in configuration:
       for name, value in parameters:
           sim.setnamed(obj, name, value)
           
    return 

# =============================================================================
# def QW_xy(params, x, y):
#     # Takes params, x, & y as arguments and returns Boolean (whether or not x,y is in QW emitting region) 
#     # Checked 2026-03-24 
#     nonemitting_thickness = 0.020e-6 
#     
#     if (params['ribbon_count'] == 0) & (params['notch_count'] == 0):
#              return True 
#          
#     def ribbon_x(x):
#         result = False 
#         for r in range(params['ribbon_count']):
#             result = result or (params['ribbon_centers'][r] - params['ribbon_widths'][r]/2 + nonemitting_thickness <= x and x <= params['ribbon_centers'][r] + params['ribbon_widths'][r]/2 - nonemitting_thickness)
#             #result = result or (params['ribbon_centers'][r] - params['ribbon_widths'][r]/2 <= x and x <= params['ribbon_centers'][r] + params['ribbon_widths'][r]/2)
#         return result 
#     
#     def notch_y(y):
#         result = False 
#         for n in range(params['notch_count']):
#             result = result or (params['notch_centers'][n] - params['notch_widths'][n]/2 - nonemitting_thickness <= y and y <= params['notch_centers'][n] + params['notch_widths'][n]/2 + nonemitting_thickness)
#             #result = result or (params['notch_centers'][n] - params['notch_widths'][n]/2 <= y and y <= params['notch_centers'][n] + params['notch_widths'][n]/2)
#         return result 
#     
#     def notch_x(x):
#         result = False 
#         for r in range(params['ribbon_count']):
#             result = result or (min_mesa_width - 2*nonemitting_thickness <= np.abs(params['ribbon_centers'][r] - x) and np.abs(params['ribbon_centers'][r] - x) <= params['ribbon_widths'][r]/2 - nonemitting_thickness)
#         return result 
#     
#     if ribbon_x(x):
#         if notch_y(y):
#             if not notch_x(x):
#                 return True 
#         else: 
#             return True 
#     return False 
# =============================================================================


def RCWA_sim(params):
    sim = lumapi.FDTD(hide=False) 
    setup(params, sim) 
    sim.save("/tmp/rcwa_session.fsp") # Throwaway file. Saving prevents a lot of issues when running solver loops with lumapi 
    sim.run()
    epsilon_inplane_superlattice = sim.getindex(params['layer_materials'][1], c/params['wavelength_range'], 1)**2
    epsilon_inplane_substrate = sim.getindex(params['layer_materials'][0], c/params['wavelength_range'], 1)**2 
    sim.eval("""
        Es = getresult("monitor","Es");
        Ep = getresult("monitor","Ep");
        """) 
    sim.switchtolayout() 
    # Skip matlabsave and instead use sim.getv() to bring variables over to python 
    Es = sim.getv("Es")
    #sim.eval("Es.Ex = []") 
    sim.eval("Es = 0;") # To free up the memory in Lumerical 
    Ep = sim.getv("Ep") 
    sim.close()
    gc.collect() 
    
    # Find vocsel size and incident power so that we can calculate A as a %
    dx = np.diff(Es['x'][:,0]).mean() if len(Es['x']) > 1 else params['period'][0]
    dy = np.diff(Es['y'][:,0]).mean() if len(Es['y']) > 1 else params['period'][1]
    dz = np.diff(Es['z'][:,0]).mean() if len(Es['z']) > 1 else params['layer_thicknesses'][1]
    dV = dx * dy * dz
    
    Z_substrate = 1 / (np.real(epsilon_inplane_substrate) * epsilon0 * c)
    A_unit_cell = params['period'][0] * params['period'][1]
    P_inc = 0.5 * (1.0 / Z_substrate) * A_unit_cell 
    
    
    A_lambda_pol = np.zeros([len(params['wavelength_range']), 2])
    # Es['E'] and E['E'] are indexed as x, y, z, lambda, angle, vector-components  
    for xi in range(len(Es['x'])):
        for yi in range(len(Es['y'])):
            for zi in range(len(Es['z'])): 
                if True: #QW_xy(params, Es['x'][xi], Es['y'][yi]):
                    # Calculate absorption 
                    Es_abs2 = (np.abs(Es['E'][xi,yi,zi,:,0,0])**2 
                               + np.abs(Es['E'][xi,yi,zi,:,0,1])**2 
                               + np.abs(Es['E'][xi,yi,zi,:,0,2])**2)
                    Ep_abs2 = (np.abs(Ep['E'][xi,yi,zi,:,0,0])**2 
                               + np.abs(Ep['E'][xi,yi,zi,:,0,1])**2 
                               + np.abs(Ep['E'][xi,yi,zi,:,0,2])**2) 
                    A_lambda_pol[:,0] += (1/2 * 2*np.pi*Es['f'] * epsilon0 * np.imag(epsilon_inplane_superlattice) * Es_abs2.reshape(-1,1)).squeeze() * dV
                    A_lambda_pol[:,1] += (1/2 * 2*np.pi*Ep['f'] * epsilon0 * np.imag(epsilon_inplane_superlattice) * Ep_abs2.reshape(-1,1)).squeeze() * dV 
    
    return A_lambda_pol / P_inc 

def FoM(params, queue=None, plot=False):
    
    A_lambda_pol = RCWA_sim(params)
    wavelengths = params['wavelength_range'] 
    
    s_arg_max = np.argmax(A_lambda_pol[:,0])
    p_arg_max = np.argmax(A_lambda_pol[:,1])
    print(f"Highest absorption is {A_lambda_pol[s_arg_max,0]} at {1e9*wavelengths[s_arg_max]:.0f} nm (s-pol) and  {A_lambda_pol[p_arg_max,0]} at {1e9*wavelengths[p_arg_max]:.0f} nm (p-pol)")
    
    
    if plot: 
        plt.plot(1e9*wavelengths, A_lambda_pol[0], label='s-pol')
        plt.plot(1e9*wavelengths, A_lambda_pol[1], label='p-pol')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Absorption')
        plt.title('Absorption vs wavelength')
        plt.show()
    
    return A_lambda_pol

    
# Test devices
params = {
          'Fourier_N' : 50, 
          'xy_mesh' : 90, 
          'z_mesh' : 55, 
          'k_mesh' : 24, 
          'wavelength_range' : np.linspace(8e-6, 9.5e-6, 100), 
          'layer_count' : 3, 
          'layer_names' : ['substrate', 'superlattice', 'superstrate'], # reciprocity plane waves are incident from first layer 
          'layer_thicknesses' : [1e-6, 0.500e-6, 0.200e-6], # SL thicknesses can be variable param 
          'layer_materials' : ["GaSb - custom", "InSb30/GaSb70 superlattice", "Au (Gold) - Palik"],
          'layer_is_etched' : [False, True, True], # whether or not to etch through each layer to make the ribbons 
          'ribbon_count' : 1, # number of nanoribbons to etch 
          'notch_count' : 1, 
          # The params below will be incorporated into 'var' as fixed or range parameters, then passed to FoM in evaluate() 
          'period' : [1.0e-6, 1.0e-6], # um 
          'ribbon_centers' : [0.5e-6], 
          'ribbon_widths' : [250e-9], 
          'notch_centers' : [0.5e-6], 
          'notch_widths' : [250e-9], 
          }

if len(params['period']) == 0:
    params['period'] = [params['wavelength_range'][-1], params['wavelength_range'][-1]]
    

A_lambda_pol = FoM(params) 

plt.plot(params['wavelength_range']*1e6, A_lambda_pol[:,0], label='s-pol')
plt.plot(params['wavelength_range']*1e6, A_lambda_pol[:,1], label='p-pol')
plt.xlabel('Wavelength (um)')
plt.ylabel("Absorbance")
plt.legend() 
plt.show() 
