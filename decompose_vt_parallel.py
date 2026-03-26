#!/usr/bin/env python
# coding: utf-8

"""
Parallel wavenumber decomposition of tangential wind (vt).
Parallelizes over (case, date) pairs using ProcessPoolExecutor.
 
Usage:
    python decompose_vt_parallel.py          # uses all CPU cores
    python decompose_vt_parallel.py --workers 4
"""

import numpy as np
import pandas as pd
import xarray as xr
import metpy.calc as mpcalc
import metpy.units as units

import sys,os, glob, re, argparse

from datetime import datetime,timedelta
from netCDF4 import Dataset
from scipy.signal import butter, filtfilt
from scipy.ndimage import map_coordinates

import wrf
import metpy

from concurrent.futures import ProcessPoolExecutor

# ── Constants ─────────────────────────────────────────────────────────────────
 
hlevels   = [500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
dr        = 1000.          # radial grid spacing (m)
dtheta    = 1.             # azimuthal resolution (deg)
Re        = metpy.constants.earth_avg_radius
 
cytrk_dir = "/home/hoadnq/twin_TCs/CyTRACK/testing_CyTRACK/domain_02/"
wrf_dir   = "/work/users/hoadnq/wrfv4/output/"
out_dir   = "/work/users/hoadnq/twin_TCs/wave_decom/NC/"            # output directory – change as needed

# ── Utility functions ─────────────────────────────────────────────────────────
 
def reconstruct_2d_field(fft_coeffs, theta_grid, wavenumber):
    """
    Reconstructs the 2-D spatial field for a single azimuthal wavenumber.
 
    Parameters
    ----------
    fft_coeffs : ndarray (n_azimuth, n_radius)
        Output of np.fft.fft along the azimuthal axis, already normalised
        by n_azimuth.
    theta_grid : ndarray (n_azimuth, n_radius)
        Azimuth in radians at every polar-grid point.
    wavenumber : int
        0 = mean (WN0), 1 = WN1, 2 = WN2, …
 
    Returns
    -------
    ndarray (n_azimuth, n_radius)
    """
    coeff_n = fft_coeffs[wavenumber, :]          # shape: (n_radius,)
 
    if wavenumber == 0:
        reconstructed = np.real(coeff_n)
        return np.tile(reconstructed, (theta_grid.shape[0], 1))
    else:
        coeff_n_2d = np.tile(coeff_n, (theta_grid.shape[0], 1))
        term = coeff_n_2d * np.exp(1j * wavenumber * theta_grid)
        return 2 * np.real(term)
 
 
def create_polar_grid_latlon(center_lon, center_lat, max_r, dr, dtheta):
    """
    Creates a polar grid and returns (lon, lat) of every grid point.
    Uses an Azimuthal-Equidistant projection to guarantee accurate distances.
 
    Returns
    -------
    lon_grid, lat_grid : ndarray (n_azimuth, n_radius)
    R                  : ndarray – radius (m)
    THETA              : ndarray – azimuth (rad)
    """
    import pyproj
 
    aeqd = pyproj.Proj(
        proj='aeqd', lat_0=center_lat, lon_0=center_lon, datum='WGS84'
    )
 
    radii    = np.arange(0, max_r, dr)
    azimuths = np.deg2rad(np.arange(0, 360, dtheta))
 
    R, THETA = np.meshgrid(radii, azimuths)
 
    X_proj = R * np.cos(THETA)
    Y_proj = R * np.sin(THETA)
 
    lon_grid, lat_grid = aeqd(X_proj, Y_proj, inverse=True)
    return lon_grid, lat_grid, R, THETA

# ── Worker function (one (case, date) pair per call) ─────────────────────────
 
def process_one_date(args):
    """
    Process a single (case, date) combination.
 
    Parameters
    ----------
    args : tuple
        (case, track_file, date, out_dir)
 
    Returns
    -------
    str  – status message
    """
    case, track_file, date, out_dir = args
 
    date_str   = date.strftime("%Y-%m-%d_%H:%M:%S")
    fmt_odate  = date.strftime('%Y%m%d%H%M')
 
    # ── Load track ────────────────────────────────────────────────────────────
    track = pd.read_csv(
        track_file,
        delim_whitespace=True,
        header=None,
        names=["latc", "lonc", "Pc", "Vmax", "Size",
               "Proci", "ROCI", "Core", "VTL", "VTU"],
    )
 
    # ── Open WRF file ─────────────────────────────────────────────────────────
    ncfile = f"{wrf_dir}{case}/wrfout_d02_{date_str}"
    if not os.path.exists(ncfile):
        return f"[SKIP] {ncfile} not found"
 
    nc   = Dataset(ncfile)
    slp  = wrf.getvar(nc, "slp")
    lats, lons = wrf.latlon_coords(slp)
    rlat = np.deg2rad(lats)
    rlon = np.deg2rad(lons)
 
    # ── Wind & pressure fields ────────────────────────────────────────────────
    uvmet  = wrf.getvar(nc, "uvmet", units="m s-1")
    ua, va = uvmet[0], uvmet[1]
    height = wrf.getvar(nc, "height_agl")
 
    u_h = wrf.interplevel(ua, height, hlevels)
    v_h = wrf.interplevel(va, height, hlevels)
 
    # ── Loop over track rows (vortex centres) ─────────────────────────────────
    saved_files = []
    for index, row in track.iterrows():
        center_lat = row['latc']
        center_lon = row['lonc']
        max_radius = row['Size'] * 1000.          # km → m
 
        rclat = np.deg2rad(center_lat)
        rclon = np.deg2rad(center_lon)
 
        dx    = Re * np.cos(rclat) * (rlon - rclon)
        dy    = Re * (rlat - rclat)
        theta = np.arctan2(dy, dx).metpy.dequantify()
 
        # Tangential wind (earth-relative)
        vt = (- u_h * np.sin(theta) + v_h * np.cos(theta)).metpy.dequantify()
 
        # Build polar grid
        lon_targets, lat_targets, R_grid, Theta_grid = create_polar_grid_latlon(
            center_lon, center_lat, max_radius, dr=dr, dtheta=dtheta
        )
 
        xy_points     = wrf.ll_to_xy(nc, lat_targets, lon_targets)
        coords_indices = np.array([xy_points[1].values, xy_points[0].values])
        n_azimuth, n_radius = R_grid.shape
 
        # ── Per-level FFT decomposition ───────────────────────────────────────
        storage_wn0, storage_wn1, storage_wn2 = [], [], []
 
        for level in hlevels:
            vt_earth = vt.sel(level=level)
            vt_flat  = map_coordinates(
                vt_earth.values, coords_indices, order=3, mode='nearest'
            )
            vt_polar = vt_flat.reshape(n_azimuth, n_radius)
 
            # Azimuthal FFT  (axis-0 = azimuthal)
            fft_coeffs = np.fft.fft(vt_polar, axis=0) / n_azimuth
 
            storage_wn0.append(reconstruct_2d_field(fft_coeffs, Theta_grid, 0))
            storage_wn1.append(reconstruct_2d_field(fft_coeffs, Theta_grid, 1))
            storage_wn2.append(reconstruct_2d_field(fft_coeffs, Theta_grid, 2))
 
        # ── Build & save xarray Dataset ───────────────────────────────────────
        ds_out = xr.Dataset(
            data_vars={
                "vt_wn0": (("level", "azimuth", "radius"), np.array(storage_wn0)),
                "vt_wn1": (("level", "azimuth", "radius"), np.array(storage_wn1)),
                "vt_wn2": (("level", "azimuth", "radius"), np.array(storage_wn2)),
            },
            coords={
                "level":   hlevels,
                "azimuth": np.degrees(Theta_grid[:, 0]),
                "radius":  R_grid[0, :] / 1000.0,    # save in km
            },
        )
 
        out_name = os.path.join(
            out_dir,
            f"decomposed_vt_{case}_core{index}_{date.strftime('%Y%m%d%H')}.nc"
        )
        ds_out.to_netcdf(out_name)
        saved_files.append(out_name)
 
    nc.close()
    return f"[OK] {case} | {date_str} → {len(saved_files)} file(s) saved"
 
 
# ── Build task list ───────────────────────────────────────────────────────────

def build_tasks(case_filter = None):
    """
    Return a list of (case, track_file, date, out_dir) tuples.
    
    Parameters
    ----------
    case_filter : list of int, optimal
        Indices into case_names to process (e.g. [5]). None = all cases.
    """
    paths      = sorted(glob.glob(cytrk_dir + "*"))
    case_names = [os.path.basename(p) for p in paths]
    
    if case_filter is not None:
        paths      = [paths[i]      for i in case_filter]
        case_names = [case_names[i] for i in case_filter]
        
    tasks = []
    for case, case_path in zip(case_names, paths):
        tracks = sorted(glob.glob(case_path + "/*"))
        dates  = [
            pd.to_datetime(re.search(r'\d{10}', p).group(), format = '%Y%m%d%H')
            for p in tracks
        ]
        for track_file, date in zip(tracks, dates):
            tasks.append((case, track_file, date, out_dir))
            
    return tasks

# ── Main ──────────────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(description = "Parallel vt wavenumber decomposition")
    parser.add_argument("--workers", type = int, default = None,
                        help = "Number of parallel workers (default: all CPU cores)")
    parser.add_argument("--cases", type = int, nargs = "+", default = [5],
                        help = "Case indices to process (default: [5]). "
                               "Pass -1 for all cases.")
    args = parser.parse_args()
    
    case_filter = None if (args.cases == [-1]) else args.cases
    tasks       = build_tasks(case_filter = case_filter)
    
    print(f"Total tasks: {len(tasks)}  |  workers: {args.workers or 'auto'}")
 
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one_date, t): t for t in tasks}
 
        for future in as_completed(futures):
            task = futures[future]
            try:
                msg = future.result()
                print(msg)
            except Exception as exc:
                case, _, date, _ = task
                print(f"[ERROR] {case} | {date} → {exc}")
 
 
if __name__ == "__main__":
    main()
