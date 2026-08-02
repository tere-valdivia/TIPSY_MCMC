import numpy as np
import emcee
import tipsy
from spectral_cube import SpectralCube as SC
from scipy.interpolate import interp1d
from sklearn.model_selection import ParameterGrid
from astropy import units as u
from astropy.io import fits
# add your path to tipsy here 
import sys
sys.path.append('/Users/mvaldivi/Documents/Science/TIPSY/')
from tipsy import *
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

def traj_fitting_space_exploration_from_dcmeans(datapoints,header,Ms_val,dist,svel,N_elements=10,theta_weight=1
                 ,vxy_ang0=None,vxy_ang0_span=None,vxy_ang0_step=0.0872
                 ,z0_lim=None,z0_step=None,vxy0_lim=None,vxy0_step=None
                 ,show_dist_plots=True,show_vel_ang=True,show_fit_cost=True,show_fit_3d=True,fit2html=False,html_path=None
                 ,add_times=True, theta_ref_trajectory = False, verbose=True):
    '''
    Code to fit infalling trajectories in PPV space to streamers.
    All below is straight up copied from the original TIPSY function traj_fitting, just changed 
    to take in data points instead of fits file and to use the distance metric defined from the data.
    
    Args:
datapoints: n x 3 numpy array with data points in observational coordinates (arcsec, arcsec, m/s) of where the streamer is suspected to be
    header: Header of the original datacube for the data points
    Ms_val ([float]): Mass of central object [solar mass]
    dist ([float*unit]): Distance to the source multiplied by its unit as an astropy.units object
    svel ([float]): Systematic radial velocity of the source [km/s]
    N_elements (Optional[int]): No. of elements to use for fitting
    theta_weight (Optional[int]): Weightage of projected angles along the streamer for distance metric, 0 means just using projected distances
    vxy_ang0 (Optional[float]): Initial angle between P.O.S. velocity and R.A. axis [radians]
    vxy_ang0_span (Optional[float]): Total span of initial angles between P.O.S. velocity and R.A. axis to use for parameter space [radians]
    vxy_ang0_step (Optional[float]): Resolution of initial angles between P.O.S. velocity and R.A. axis to use for parameter space, default is ~5 degrees [radians]
    z0_lim (Optional[tuple]): Min. and max. value of z0 (rel. radial dist.) to generate trajectories [au]
    z0_step (Optional[float]): Resolution for z0 to generate trajectories [au]
    vxy0_lim (Optional[tuple]): Min. and max. value of vxy0 (speed in RA/Decl. plane) to generate trajectories [km/s]
    vxy0_step (Optional[float]): Resolution for z0 to generate trajectories [km/s]    
    show_dist_plots (Optional[bool]): Whether to show plots of projected distances, angles, and final distance metrics (independent parameter for fitting)
    show_vel_ang (Optional[bool]): Whether to show estimation of initial orientation of P.O.S. velocity
    show_fit_cost (Optional[bool]): Whether to show plot with fitting fraction (and deviation) as a function of free parameters
    show_fit_3d (Optional[bool]): Whether to show 3D plot original point cloud, points used for fitting and final trajectory 
    fit2html (Optional[bool]): Whether to save 3D fitting plot to .html file
    html_path (Optional[str]): Path to save 3D fitting plot to .html file
    add_times (Optional[bool]): Whether to add final times of trajectories (infalling time scales) to
    the param_grid2 (dataframe with all initial parameter combinations, see output)
    theta_ref_trajectory (Optional[bool]): Whether to compute a new theta (polar angle) reference for each theoretical trajectory. It can affect dist. metric values of trajectories and thus, interpolation. 
    verbose (Optional[bool]): Whether to display miscellaneous text outputs 
    
    Returns:
    param_grid2: Pandas DataFrame with all the parameter combinations used and corresponding deviations
    traj_m: Trajectory corresponding to the best fit
    [traj_comp_m,pcmeans,pcstds]: List of three 3xN_elements numpy arrays storing values for interpolated traj_m, means of streamer, and standard deviations of streamer
'''
    d_means, x_t, y_t, vz_t, x_err, y_err, vz_err = datapoints
    pcmeans = np.array([x_t,y_t,vz_t])
    pcstds = np.array([x_err,y_err,vz_err])

    # From here on, its just the original TIPSY code with slight modifications: https://github.com/AashishGpta/TIPSY

    ## Initial distance (arcseec to au). Note: errors are not accounted here
    dist = (dist).to(u.pc) # can handle any distance units
    x0_as = pcmeans[0][-1] #*u.arcsec
    y0_as = pcmeans[1][-1] #*u.arcsec
    x0 = arcsec2au(x0_as,dist)
    y0 = arcsec2au(y0_as,dist) #(x0_as.to(u.radian).value*dist).to(u.au).value
    ## Initial L.O.S. velocity (km/s)
    vz0 = pcmeans[2][-1]*1e-3-svel
    print("Initial x0, y0 and vz0:",x0,y0,vz0)

    print("Now for the free parameters:")
    ### Setting grid for free parameters
    ## Initial offset in radial distance
    if z0_step == None:  # By defualt use beam size as z0 resolution
        try:
            bsize = header['BMAJ']   # Beam size (Major axis)
        except KeyError:
            print("BMAJ field doesn't exist in fits header, trying to use pixel size (CDELT2)")
            bsize = np.abs(header['CDELT2'])
        bsize_au = arcsec2au(bsize*60*60,dist)  # Beam size in au
        z0_step = np.abs(bsize_au)
    if z0_lim == None:
        dev = (np.abs(x0)+np.abs(y0))/2   # Typical vel. deviation in R.A./Dec.
        dev_z = (-1*np.sign(vz0))*dev   # Expected vel. deviation in z
        d_dev_z = max(20*z0_step,3000)  # Range of values to test
        z0_min=dev_z-d_dev_z
        z0_max=dev_z+d_dev_z
    else:
        z0_min=z0_lim[0]
        z0_max=z0_lim[1]
    z0_range = np.arange(z0_min,z0_max,z0_step)
    if verbose:
        print("Min., max. and step for initial radial sep. (z0):",z0_min,z0_max,z0_step)

    ## Initial offset in P.O.S speed
    if vxy0_step == None:  # By defualt use spectral resolution as vxy0 resolution
        cwidth = header['CDELT3']
        vxy0_step = np.abs(cwidth)
    if vxy0_lim == None:
        dev_vxy = np.abs(vz0)*(2**0.5)    #Rough estimate of vxy0 
        d_dev_vxy = max(20*vxy0_step,4)      # No good reason to choose 20 or 4 :P
        vxy0_min=max(0,dev_vxy-d_dev_vxy)  #vxy0_min always should be positive
        vxy0_max=dev_vxy+d_dev_vxy
    else:
        vxy0_min=vxy0_lim[0]
        vxy0_max=vxy0_lim[1]
    vxy0_range = np.arange(vxy0_min,vxy0_max,vxy0_step)
    if verbose:
        print("Min., max. and step for initial speed in P.O.S. (vxy0):",vxy0_min,vxy0_max,vxy0_step)

    ## Orientation of P.O.S velocity wrt R.A. axis, Changed by Tere Valdivia-Mena
    dx = pcmeans[0][-2]-pcmeans[0][-1]    # Initial change in R.A.
    dy = pcmeans[1][-2]-pcmeans[1][-1]    # Initial change in Decl.
    
     #Angle between total x-y velocity and velocity in x (R.A.), main estimate of vxy0_ang
    if vxy_ang0 is None: # changed == to is
        vxy_ang0_est = np.arctan2(dy,dx)
    else: 
        vxy_ang0_est = vxy_ang0

    if vxy_ang0_span == None:    # If span for intial orientation angles is not provided, estimate it using error in estimated angle
        e_x = pcstds[0][-2]  # Error in dx
        e_y = pcstds[1][-2]  # Error in dy
        r = dy/dx
        e_r = r*np.sqrt((e_x/dx)**2 + (e_y/dy)**2)
        vxy_ang0_error = np.abs(e_r/(1+r**2)) # Error in vxy_ang0
        vxy_ang0_span = 2*vxy_ang0_error
    r1 = np.arange(vxy_ang0_est,vxy_ang0_est+(vxy_ang0_span/2),vxy_ang0_step)
    r2 = np.arange(vxy_ang0_est,vxy_ang0_est-(vxy_ang0_span/2),-vxy_ang0_step)
    vxy_ang0_range = np.concatenate([r2[::-1][:-1],r1])
    print(vxy_ang0_range)

    if verbose:
        print("Min., max. and step for initial directions in P.O.S. (vxy_ang0):",vxy_ang0_range[0],vxy_ang0_range[-1],vxy_ang0_step)
    if show_vel_ang: # To visualise vxy0_ang values on projected streamer
        i1 = 0
        i2 = 1
        labels = ['R.A. offset [arcsec]','Decl. offset [arcsec]','L.O.S. Vel. offset']  # Just put an array to make it easier in future
        plt.errorbar(x = pcmeans[i1],y = pcmeans[i2],xerr = pcstds[i1],yerr = pcstds[i2]
                     ,fmt = 's-',alpha = 1, label = 'Means')
        plt.plot(x0_as,y0_as,'ko',alpha = 1)
    #     plt.plot(x_smoothed,y_smoothed,'P-',label='Spline fit')
        plt.quiver(pcmeans[0][-1],pcmeans[1][-1],np.cos(vxy_ang0_est),np.sin(vxy_ang0_est),
                   angles='xy',scale_units = 'dots',scale=0.01,label='Initial direction',zorder=5)
        for vxy_ang0_i in vxy_ang0_range:
            plt.quiver(pcmeans[0][-1],pcmeans[1][-1],np.cos(vxy_ang0_i),np.sin(vxy_ang0_i),
                       angles='xy',scale_units = 'dots',scale=0.01,zorder=5,alpha=0.3)  # Maybe something is wrong here, because spacing sometimes look unequal..
        plt.xlabel(labels[i1])
        plt.ylabel(labels[i2])
        plt.title("Discretization of streamer and possible initial orientations")
        plt.legend()
        plt.show()

    ## Creating parameter grid
    param_grid = {'x0':[x0],'y0':[y0],'z0':z0_range
                  ,'vxy_ang0':vxy_ang0_range,'vxy0':vxy0_range,'vz0':[vz0]
                  ,'Ms_val':[Ms_val]}
    param_grid2 = pd.DataFrame(ParameterGrid(param_grid))
    param_grid2['fit_fraction'] = np.nan
    param_grid2['deviation'] = np.nan
    if add_times:
        param_grid2['final_time'] = np.nan # to store infalling time scales
        
    N_params = len(param_grid2)
    if verbose:
        print("Total no. of parameter combinations:",N_params)

    ## Fitting finally (NOTE: Can be optimized to not check for unneccessary values)
    max_frac = -1
    min_dev = np.inf
    for i in range(N_params):

        if i==0:
            st = time.time()   # To get a time estimate for this loop

        # reformatting initial parameters and computing trajectories
        p = param_grid2.loc[i]
        vx0 = p['vxy0']*np.cos(p['vxy_ang0'])
        vy0 = p['vxy0']*np.sin(p['vxy_ang0'])
        rt,vt,rt_spherical,vt_spherical,times = falling_trajectory(p['x0'],p['y0'],p['z0'],vx0,vy0,p['vz0'],Ms_val
                                                             ,ang_range = 1.9*np.pi,ang_step = np.pi/200,verbose=False)
        if add_times:
            ftimes = times[np.argsort(np.abs(rt_spherical[0]))]  # Orders times on the basis of how close the particle is to the star, sort of final times we want
            ftime = ftimes[~np.isnan(ftimes)][0]  # Select the first non nan final time, nan times are usually for hyperbolic trajectories
            param_grid2.loc[i,'final_time'] = ftime  # storing the timestamp for closest approach as infalling timescale
            # param_grid2.loc[i,'final_time'] = times[-1]  # storing the final timestamp as infalling timescale, can be optimised..
        x_t_full = au2arcsec(rt[0],dist)  # R.A. in arcsec
        y_t_full = au2arcsec(rt[1],dist)  # Decl. in arcsec
        vz_t_full = (vt[2]+svel)*1e3   # R.V. in m/s
        pctraj_full = np.array([x_t_full,y_t_full,vz_t_full]) # trajectory for full orbit in observational coordinates

        # Selecting portion of trajectory within the area of streamer
        # Note: I did not put a condition on vel. because it is not used in dist. metric and dynamics may change
        # Modification 1: using the datapoints instead of the point cloud to define the area of streamer.
        iaxis = 0
        con1 = (pctraj_full[iaxis]>=min(x_t-2*x_err)) & (pctraj_full[iaxis]<=max(x_t+2*x_err))  # Putting some error margin here, otherwise interpolation can be very bad for trajectories which just come within the field
        iaxis = 1
        con2 = (pctraj_full[iaxis]>=min(y_t-2*y_err)) & (pctraj_full[iaxis]<=max(y_t+2*y_err))
        con_pc = con1&con2
        con_pc[np.argmax(~con_pc):]=False # This is to remove trajectory section which reappears within FOV, very important to make interpolation work well
        pctraj=pctraj_full[:,con_pc]


        if pctraj.shape[1]>2:   #check if some traj. even left within the streamer field       
            # Interpolating trajectory to same "distances" (used as independent param.) as means
            # Note: Sometimes the function extrapolates too (for smaller distances), not always a good one
            r_t,theta_t = spherical_coords(pctraj[0],pctraj[1])
            if theta_ref_trajectory: # Calculating new theta_ref for each traj, can be used to avoid weirdnesses
                r_percentile_thresh = 100/N_elements # Modification 2: calculate the theta_ref.
                theta_ref_t = np.median(theta_t[r_t<np.percentile(r_t,r_percentile_thresh)])
                theta2_t = np.pi-np.abs(np.pi-np.abs(theta_t-theta_ref_t))  
            else: 
                # Using same theta_ref as observations by default. It is probably better to have a consistent reference everywhere and keep it simple. 
                # Overall also seems better than individual ref. for each trajectory (and also ref. from means) but still may not always be the the best solution. 
                theta2_t = np.pi-np.abs(np.pi-np.abs(theta_t-theta_ref))  # Using same theta_ref as observations
            # theta_ref_t = np.median(theta_t[r_t<r_thresh])  # Calculating new theta_ref but using same r_thresh as streamer, discarded because it returns nan for trajectories which don't come close enough
            # theta2_t = np.pi-np.abs(np.pi-np.abs(theta_t-theta_ref_means))  # Using same theta_ref as means, right thing to do?
            d_t = r_t*np.sqrt(1+(theta_weight*theta2_t)**2)
            p = np.argsort(d_t)
        #     spx = UnivariateSpline(d_t[p], pctraj[0][p],k=kspline)#, s=s)
        #     spy = UnivariateSpline(d_t[p], pctraj[1][p],k=kspline)#, s=s)
        #     spz = UnivariateSpline(d_t[p], pctraj[2][p],k=kspline)#, s=s)
            spx = interp1d(d_t[p], pctraj[0][p],fill_value='extrapolate',kind='slinear')#, s=s)
            spy = interp1d(d_t[p], pctraj[1][p],fill_value='extrapolate',kind='slinear')#, s=s)
            spz = interp1d(d_t[p], pctraj[2][p],fill_value='extrapolate',kind='slinear')#, s=s)
            traj_comp = np.array([spx(d_means),spy(d_means),spz(d_means)])
        else:
            traj_comp = np.full(pcmeans.shape,np.nan)  # Just nan array of same shape

        ## For goodness-of-fit estimates
        res = pcmeans-traj_comp  
        res_norm_sq = (res/pcstds)**2
        fit_frac = np.sum((res_norm_sq)<1)/3/N_elements  # Fraction of observational value consistent with the trajectory
        chi2 = np.sum(res_norm_sq)
        param_grid2.loc[i,'deviation'] = chi2  # In principle, deviation doesn't have to be chi2, maybe a median dev. be better?
        param_grid2.loc[i,'fit_fraction'] = fit_frac
        # To store the best trajectory on the fly, maybe outdated now because I store everything
        if param_grid2.loc[i,'fit_fraction']>max_frac:   
            i_m=i
            max_frac=param_grid2.loc[i,'fit_fraction']
            min_dev=param_grid2.loc[i,'deviation']
            traj_m = [rt,vt,rt_spherical,vt_spherical] # traj_m, to be returned, as positions (rt) and velocities (vt) in both cartesian and spherical coordinates
            traj_comp_m = traj_comp
        elif param_grid2.loc[i,'fit_fraction']==max_frac:
            if param_grid2.loc[i,'deviation']<min_dev:
                i_m=i
                min_dev=param_grid2.loc[i,'deviation']
                traj_m = [rt,vt,rt_spherical,vt_spherical]
                traj_comp_m = traj_comp

        if i==0:
            exec_time = time.time() - st   # To get a time estimate
            print("Expected time for fitting [mins]:",np.round(exec_time*N_params/60,1))
    param_m = param_grid2.iloc[i_m]
      
    if verbose:
        print("Time taken for fitting [mins]:",np.round((time.time()-st)/60,1))
        print("The best solution (combination index {}):\n Fraction of fitted values: {:.2f}\n Deviation: {:.2f}\n z0 [au]: {:.2f}\n vxy0 [km/s]: {:.2f} and vxy_ang0 [degree]: {:.4f} (vx0: {:.2f} and vy0: {:.2f})"
              .format(i_m,param_m['fit_fraction'],param_m['deviation'],param_m['z0'],param_m['vxy0'],param_m['vxy_ang0']*180/np.pi
                      ,param_m['vxy0']*np.cos(param_m['vxy_ang0']),param_m['vxy0']*np.sin(param_m['vxy_ang0'])))

                
    if show_fit_cost:
        # Note: maybe a imshow like triangular figure (with 3 axis) will be nice
        ## Distribution of fitting fraction wrt vxy_ang0
        ang_means = param_grid2.groupby('vxy_ang0').mean()
        ang_stds = param_grid2.groupby('vxy_ang0').std()
        plt.errorbar(ang_means.index*180/np.pi,ang_means.fit_fraction,yerr=ang_stds.fit_fraction,fmt='o')
        plt.plot(param_m['vxy_ang0']*180/np.pi,param_m.fit_fraction,'rx',markersize=15, label='Best fit')
        plt.xlabel('Orientation of initial P.O.S. velocity (vxy_ang0) [degree]')
        plt.ylabel('Mean fitting fraction')
        plt.legend()
        # plt.title("Goodness of fit as a function of initial direction")
        plt.show()

        grid2display = param_grid2[param_grid2['vxy_ang0']==param_m['vxy_ang0']]
        cmap = plt.cm.viridis
        cmap.set_bad(color='white')  # Nan values will show as red
        
        ## Distribution of fitting fraction wrt z0 and vxy0
        cfield = 'fit_fraction'
        pivot_df = grid2display.pivot(index='z0', columns='vxy0', values=cfield)
        pivot_df.sort_index(axis=0,inplace=True,ascending=True)
        pivot_df.sort_index(axis=1,inplace=True,ascending=True)
        plt.imshow(pivot_df,extent = [pivot_df.columns[0]-vxy0_step/2,pivot_df.columns[-1]+vxy0_step/2
                                      ,pivot_df.index[-1]+z0_step/2,pivot_df.index[0]-z0_step/2],aspect='auto',cmap=cmap)
        plt.plot(param_m['vxy0'],param_m['z0'],'rx',markersize = 15, label='Best fit')
        plt.ylabel('Initial radial distance (z0) [au]')
        plt.xlabel('Initial POS speed (vxy0) [km/s]')
        plt.legend()
        plt.colorbar(label='Fitting fraction')
        # plt.title("Goodness of fit as a function of initial direction")
        plt.show()

        ## Distribution of deviation (loglog scaled) wrt z0 and vxy0
        cfield = 'loglogdev'
        grid2display['loglogdev'] = np.log10(np.log10(grid2display['deviation']))
        pivot_df = grid2display.pivot(index='z0', columns='vxy0', values=cfield)
        pivot_df.sort_index(axis=0,inplace=True,ascending=True)
        pivot_df.sort_index(axis=1,inplace=True,ascending=True)
        plt.imshow(pivot_df,extent = [pivot_df.columns[0]-vxy0_step/2,pivot_df.columns[-1]+vxy0_step/2
                                      ,pivot_df.index[-1]+z0_step/2,pivot_df.index[0]-z0_step/2],aspect='auto',cmap=cmap.reversed())
        plt.plot(param_m['vxy0'],param_m['z0'],'rx',markersize = 15, label='Best fit')
        plt.ylabel('Initial radial distance (z0) [au]')
        plt.xlabel('Initial POS speed (vxy0) [km/s]')
        plt.legend()
        plt.colorbar(label='log(log(deviation))')
        plt.show()
        
    # Modification 3: 3D plot with just the means and trajectory, without the point cloud, as this for ACORNS does not exist.
    if show_fit_3d:
        x_t = au2arcsec(traj_m[0][0],dist)
        y_t = au2arcsec(traj_m[0][1],dist)
        vz_t = (traj_m[1][2]+svel)*1e3
        pctraj_m = np.array([x_t,y_t,vz_t]) # trajectory in same format as everything else
        fit_3d_plot_no_pcloud(pcmeans=pcmeans,pcstds=pcstds,traj=pctraj_m,traj_interp=traj_comp_m,fit2html=fit2html,html_path=html_path)          
    
    return(param_grid2,traj_m,[traj_comp_m,pcmeans,pcstds])

def fit_3d_plot_no_pcloud(pcmeans=None,pcstds=None,flux=None,traj=None,traj_interp=None
                ,fit2html=False,html_path=None,fit2png=False,png_path=None,camera_vec=[1.5,1.5,1.5],dtick=None):

    '''
    Code to visualize the observational curve together with the theoretical curve for streamers in PPV space.
    Straight up copied from the original TIPSY function fit_3d_plot, just changed to not include the point cloud in the plot, as this for ACORNS does not exist.
    
    Args:
    pcmeans: 3xN_elements array containing coordinates of the means of points in streamer
    pcstds: 3xN_elements array containing coordinates of the standard deviations of points in streamer
    flux: N_elements long array containing flux values for observed streamer (used for colour)
    traj: 3xN1 array containing coordinates for a theoretical trajectory
    traj_interp: 3xN_elements array containing interpolated (or some extrapolated) coordinates for a theoretical trajectory
    fit2html: Whether to save 3D fitting plot to an interactive .html file    
    html_path: Path to save 3D fitting plot to .html file    
    fit2png: If True, save PPV diagram to a static .png file.
    png_path: Path to save 3D fitting plot to .png file    
    camera_vec: An array of length three to set camera position, see plotly.com/python/3d-camera-controls/ for more details
    dtick: Spacing between ticks on x and y axes, in axes units. Added for consistent figures.
    '''
    
    data2plt = []  # All things to plot will be added to this list
    
    if traj is not None:
        data2plt+=[go.Scatter3d(x=traj[0], y=traj[1], z=traj[2],mode='lines',name='Trajectory',
                       marker=dict(size=7,color='black',opacity=1)
                       )]

    if traj_interp is not None:
        data2plt+=[go.Scatter3d(x=traj_interp[0], y=traj_interp[1], z=traj_interp[2],mode='markers',name='Trajectory (interpolated)',
                       marker=dict(size=7,color='black',opacity=1)
                       )]
        
    if pcmeans is not None:
        if pcstds is None:
            pcstds = np.zeros(pcmeans.shape)
        data2plt+=[go.Scatter3d(x=pcmeans[0], y=pcmeans[1], z=pcmeans[2],mode='lines+markers',name='Means',
                       error_x=go.scatter3d.ErrorX(array=pcstds[0],width=2),
                       error_y=go.scatter3d.ErrorY(array=pcstds[1],width=2),
                       error_z=go.scatter3d.ErrorZ(array=pcstds[2],width=2),
                       marker=dict(size=9,color='red',opacity=0.8,symbol='square')
                   )]

    layout = go.Layout(scene=dict(xaxis_title='R.A. offset [arcsec]',
                          yaxis_title='Decl. offset [arcsec]',
                          zaxis_title='L.O.S. velocity [m/s]',
                          xaxis_range=[np.nanmin(pcmeans[0]), np.nanmax(pcmeans[0])],
                          yaxis_range=[np.nanmin(pcmeans[1]), np.nanmax(pcmeans[1])],
                          zaxis_range=[np.nanmin(pcmeans[2]), np.nanmax(pcmeans[2])],
                          aspectmode='cube'),
    #                       margin=dict(l=0, r=0, b=0, t=0), 
               margin=dict(l=0, r=0, b=0, t=0), showlegend=True,
               )

    fig = go.Figure(data=data2plt, layout=layout)
    
    camera = dict(up=dict(x=0, y=0, z=1),center=dict(x=0, y=0, z=0),
                  eye=dict(x=camera_vec[0], y=camera_vec[1], z=camera_vec[2]))
    fig.update_layout(scene_camera=camera)  # Set viewing angle
    fig.update_layout(legend_orientation="h")
    if dtick:
        fig.update_layout(scene = dict(xaxis = dict(dtick =dtick),yaxis = dict(dtick = dtick)))

    if fit2html:  # To save plot as interactive html file
        if html_path == None:
            html_path = str(input("Path to store html 3D plot:"))
        fig.write_html(html_path, include_plotlyjs='cdn')
    
    if fit2png:  # To save plot as static png file, useful for publications
        if png_path == None:
            png_path = str(input("Path to store png plot:"))
        fig.write_image(png_path,format='png',scale=3)
        
    fig.show()

# definition of the log-probability function for MCMC, which includes the log-likelihood and the log-prior


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

