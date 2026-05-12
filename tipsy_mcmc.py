import numpy as np
import emcee
import tipsy
from spectral_cube import SpectralCube as SC
from scipy.interpolate import interp1d
# add your path to tipsy here 
# import sys
# sys.path.append('/your/path/to/TIPSY/')
'''
What are our data points?
- x: observed position in RA
- y: observed position in Dec
- z: observed position in RV
Think if we want to compare the datapoints to model in arcseconds or in au.
We need to define the model and the data with respect to a distance metric
'''

def distance_metric(pointcloudcoords, N_elements=10, theta_weight=1):
    pcd_r,pcd_theta = tipsy.spherical_coords(pointcloudcoords[0],pointcloudcoords[1])
    r_percentile_thresh = 100/N_elements  # No. of points to consider for ref. angle
    r_thresh = np.percentile(pcd_r,r_percentile_thresh)  # threshold radial dist. to select close enough points
    theta_ref = np.median(pcd_theta[pcd_r<r_thresh])  # Median theta angle for close enough points
    # cyclic deviation upto 180
    pcd_theta2 = np.pi-np.abs(np.pi-np.abs(pcd_theta-theta_ref))
    pcd = pcd_r*np.sqrt(1+(theta_weight*pcd_theta2)**2)  # Final dist. metric values
    return pcd, theta_ref

def cube_to_pointcloud(streamer_cube):
    # taken directly from tipsy (Aashish)
    # we need this to return the same as the falling_trajectory_model but with data (in offset arcsec)
    ## Cube to point cloud    
    pcloud = np.array(streamer_cube)
    rms_mask = ~np.isnan(pcloud)
    flux = pcloud[rms_mask]
    pcinds = np.indices(pcloud.shape)#.shape   #Array of indices of all points in streamer point cloud
    pcz = pcinds[0][rms_mask]    #Individual indices for non nan values (high enough SNR)
    pcy = pcinds[1][rms_mask]
    pcx = pcinds[2][rms_mask]   
    pccoords = np.array(streamer_cube.wcs.pixel_to_world_values(np.array([pcx,pcy,pcz]).T)).T   # Indices to FITS coords
    # converting to offset coords
    pccoords[0] = (pccoords[0]-streamer_cube.header['CRVAL1'])*60*60   # Coords in units of PPV diagram, R.A. offset
    pccoords[0] = pccoords[0]*np.cos(streamer_cube.header['CRVAL2']*np.pi/180) # cos(dec) correction required to get physical offsets in R.A. direction
    pccoords[1] = (pccoords[1]-streamer_cube.header['CRVAL2'])*60*60   # Decl. offset
    pccoords[2] = (pccoords[2]-streamer_cube.header['CRVAL3']) # L.O.S. velocity   

    return pccoords, flux # arcsec, arcsec, m/s, flux

def transform_cube_into_ydata(streamer_cube, N_elements=10, theta_weight=1):
    # taken directly from tipsy (aashish)
    # we need this to return the same as the falling_trajectory_model but with data (in offset arcsec)
    ## Cube to point cloud    
    pccoords, flux = cube_to_pointcloud(streamer_cube)
    ## Distance metric (pcd, independent parameter)
    pcdistance, theta_ref = distance_metric(pccoords, N_elements=N_elements, theta_weight=theta_weight)  # Final dist. metric values

    ## Point cloud to array of weighted-mean values (a curve-like representation)
    pcmeans = np.zeros((3,N_elements))  # intialize "empty" array
    pcstds = np.zeros((3,N_elements)) 
    b_per = np.linspace(0,100,N_elements+1)  # Array of percentile values to break the array into
    pars = np.array([np.percentile(pcdistance,per) for per in b_per])   # pcd values at percentile values 
    for i in range(N_elements):
        dinds = (pcdistance>pars[i]) & (pcdistance<=pars[i+1])  # Identify points in given distance range
        pcmeans[:,i] = np.average(pccoords.T[dinds],axis=0,weights=flux[dinds])  # Add intensity-weighted means    
        pcstds[:,i] = np.sqrt(np.average((pccoords.T[dinds]-pcmeans[:,i])**2,axis=0,weights=flux[dinds])) # Add intensity-weighted standard deviations
    r_means,theta_means = tipsy.spherical_coords(pcmeans[0],pcmeans[1])
    theta2_means = np.pi-np.abs(np.pi-np.abs(theta_means-theta_ref)) 
    d_means = r_means*np.sqrt(1+(theta_weight*theta2_means)**2)  # dist. metric for mean points, to be used for final fitting

    # check if we need the radial and theta coordinates for anything
    x_t = pcmeans[0]
    y_t = pcmeans[1]
    vz_t = pcmeans[2] # m/s

    x_err = pcstds[0]
    y_err = pcstds[1]
    vz_err = pcstds[2]  # in m/s

    return d_means, x_t, y_t, vz_t, x_err, y_err, vz_err  # distance metric, arcsec, arcsec, m/s


def find_initial_conditions(x_t, y_t, vz_t, x_err, y_err, vz_err, svel):
    # Placeholder function to find initial conditions

    ## Initial distance (arcseec to au). 
    x0_as = x_t[-1] #*u.arcsec
    y0_as = y_t[-1] #*u.arcsec
    vz0 = vz_t[-1]*1e-3 - svel # in km/s

    # intial position uncertainties (arcsec to au)
    x0_as_err = x_err[-1] #*u.arcsec
    y0_as_err = y_err[-1] #*u.arcsec
    vz0_err = vz_err[-1]*1e-3 #*u.km/u.s

    return [x0_as, y0_as, vz0], [x0_as_err, y0_as_err, vz0_err]


def falling_trajectory_model(d_means, theta, params_fixed, pccoords, theta_weight=1):
    '''
    Returns the falling trajectory model in observational coordinates (arcsec, arcsec, m/s).
    Distance metric is defined from the data, so the parameters pccords and theta_ref come from the data 
    (run distance_metric on data first).
    
    :param d_means: distance metric values at which to evaluate the model (from data)
    :param theta: parameters to fit (z0, vxy0, vxy_ang0)
    :param params_fixed: fixed parameters (x0, y0, vz0, Ms_val, dist, svel)
    :param pccoords: point cloud coodinates from data (to define distance metric theta_ref and avoid reapearing trajectory sections)
    check: do we need N_elements here?
    '''

    # r_percentile_thresh = 100/N_elements

    z0, vx0, vy0 = theta # free parameters
    x0, y0, vz0, Ms_val, dist, svel = params_fixed # fixed parameters
    rt,vt,rt_spherical,vt_spherical,times = tipsy.falling_trajectory(x0,y0,z0,vx0,vy0,vz0,
                                                                     Ms_val,verbose = False)
    x_t_full = tipsy.au2arcsec(rt[0],dist)  # R.A. in arcsec
    y_t_full = tipsy.au2arcsec(rt[1],dist)  # Decl. in arcsec
    # print(vt[2])
    vz_t_full = (vt[2]+svel)*1e3   # R.V. in m/s
    pctraj_full = np.array([x_t_full,y_t_full,vz_t_full]) # trajectory for full orbit in observational coordinates
    # Selecting portion of trajectory within the area of streamer (from Aashish's tipsy distance metric code)
    # iaxis = 0
    # con1 = (pctraj_full[iaxis]>=min(pccoords[iaxis])) & (pctraj_full[iaxis]<=max(pccoords[iaxis]))
    # iaxis = 1
    # con2 = (pctraj_full[iaxis]>=min(pccoords[iaxis])) & (pctraj_full[iaxis]<=max(pccoords[iaxis]))
    # con_pc = con1&con2
    # con_pc[np.argmax(~con_pc):]=False # This is to remove trajectory section which reappears within FOV, very important to make interpolation work well
    # pctraj=pctraj_full[:,con_pc]
    pctraj=pctraj_full
    _pcd, theta_ref = distance_metric(pccoords, theta_weight=theta_weight)  # recalculate distance metric for data points

    if pctraj.shape[1]>2:   #check if some traj. even left within the streamer field    
        # Interpolating trajectory to same "distances" (used as independent param.) as means
        # Note: Sometimes the function extrapolates too (for smaller distances), not always a good one
        r_t,theta_t = tipsy.spherical_coords(pctraj[0],pctraj[1])
        theta2_t = np.pi-np.abs(np.pi-np.abs(theta_t-theta_ref))  # Using same theta_ref as observations
        d_t = r_t*np.sqrt(1+(theta_weight*theta2_t)**2)
        p = np.argsort(d_t)
        # here we interpolate the trajectory to obtain a function of the distance metric
        spx = interp1d(d_t[p], pctraj[0][p],fill_value='extrapolate',kind='slinear')#, s=s)
        spy = interp1d(d_t[p], pctraj[1][p],fill_value='extrapolate',kind='slinear')#, s=s)
        spz = interp1d(d_t[p], pctraj[2][p],fill_value='extrapolate',kind='slinear')#, s=s)
        traj_comp = np.array([spx(d_means),spy(d_means),spz(d_means)])
    else:
        print("Warning: Trajectory exited data field too quickly, returning NaN array.") 
        traj_comp = np.full((3,len(d_means)),np.nan)  # Just nan array of same shape
    
    return traj_comp


# Log-likelihood: assume Gaussian errors
def log_likelihood(theta, d_means, y, yerr, params_fixed, pccoords):
    # remember: for falling trajectory, the fixed parameters are the stellar mass, x0, y0 and vz0
    # thus, y must be a numpy array 
    y_model = falling_trajectory_model( d_means, theta, params_fixed=params_fixed, 
                                       pccoords=pccoords)
    lnx = -0.5 * np.sum(((y - y_model) / yerr) ** 2)
    return lnx if np.isfinite(lnx) else -np.inf

def get_total_time(params, Ms):
    rt,vt,rt_spherical,vt_spherical,times = tipsy.falling_trajectory(params[0],params[1],params[2],
                                                                     params[3],params[4],params[5],
                                                                     Ms,verbose = False)
    ftimes = times[np.argsort(np.abs(rt_spherical[0]))] 
    # Select the first non nan final time, nan times are usually for hyperbolic trajectories
    ftime = ftimes[~np.isnan(ftimes)][0]  
    return ftime



# Uninformative (flat) priors

