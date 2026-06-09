import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as cons
import scipy
from scipy.optimize import curve_fit
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from . import Units
from . import MCSpec
from . import run_multiple_jobs
from . import Image

def blackbody(frequency,temperature):
    '''Calculate the blackbody spectrum for a given frequency and temperature.'''
    h = cons.h.cgs.value
    kB = cons.k_B.cgs.value
    c = cons.c.cgs.value
    return (2*h*frequency**3/c**2) / (np.exp(h*frequency/(kB*temperature)) - 1)

def fit_blackbody(freqs,luminosities):
    '''Fit a blackbody to the given spectra.'''
    fit_temp, fit_cov = scipy.optimize.curve_fit(blackbody,freqs,luminosities,p0=1e5)
    return fit_temp[0]

def plot_spec_ratio(image, ax=None, labels=None, plot_blackbody=False, temperature=None,area=None, MC_spec=None, MC_spec_args={}, MC_labels=None,freq_units='eV',lum_units='erg'):
    if not isinstance(image, list):
        image = [image]
    
    if labels is not None and not isinstance(labels, list):
        raise TypeError("labels must be a list of strings, not a single string")
    
    if labels is not None and len(labels) != len(image):
        raise ValueError(f"labels must have same length as image list ({len(labels)} != {len(image)})")
    
    #establish Hz as the normal frequency baseline and just apply a frequency_unit throughout
    if freq_units == 'eV':
        frequency_unit = Units.h_ev
        ax.set_xlabel('eV')
    else:
        frequency_unit = 1
        ax.set_xlabel('Hz')
    #establish erg as the normal energy baseline and just apply a lum_unit throughout
    if lum_units == 'eV':
        lum_unit = 1/Units.eV_2_erg
        ax.set_label('eV')
    else:
        lum_unit =1
        ax.set_label('erg')

    for i in range(len(image)):
        frequencies = image[i].frequencies
        L = image[i].get_luminosity()
        if (ax is None) and i==0:
             ax = plt.gca()

    if plot_blackbody:
        if temperature is None:
            temperature = fit_blackbody(frequencies,L)
        bb_freq = np.logspace(np.log10(min(frequencies)), np.log10(max(frequencies)), 100)
        bb_flux = blackbody(bb_freq, temperature)
    
        ax.plot(bb_freq*frequency_unit, (L*frequencies*lum_unit)/bb_flux*bb_freq*lum_unit*area, label='Blacklight/Blackbody (T={0:.2e} K)'.format(temperature))
    

    if MC_spec ==None and len(MC_spec_args)==5:
        MC_spec = MCSpec(MC_spec_args['directory'],MC_spec_args['nproc'],nfreq=MC_spec_args['nfreq'],emin=MC_spec_args['emin'],emax=MC_spec_args['emax'])
    elif MC_spec ==None and len(MC_spec_args)==2:
            MC_spec = MCSpec(MC_spec_args['directory'],MC_spec_args['nproc'])
    if MC_spec != None:
        if not isinstance(MC_spec, list):
            MC_spec = [MC_spec]
        raise UserWarning('correct handling of frequencies for ratio not done yet')
        for i in range(len(MC_spec)):
            ax.errorbar(MC_spec[i].freq*frequency_unit*1e3/Units.h_ev,(L*frequencies*lum_unit)/MC_spec[i].lum*lum_unit,yerr=MC_spec[i].lum_err, label=MC_labels[i] if MC_labels is not None else 'Blacklight/MC Spectrum {0}'.format(i),marker='s')
            
            
    ax.legend()
    ax.set_xlabel('$\\nu$ ['+freq_units+']')
    return ax



def plot_spectra(image, ax=None, labels=None, plot_blackbody=False, temperature=None,area=None, MC_spec=None, MC_spec_args={}, MC_labels=None,freq_units='eV',lum_units='erg'):
    '''Plot the spectra from one or multiple Image objects. Optionally also plot a blackbody spectrum and/or Monte Carlo spectra with error bars.
    
    Inputs:
    - image: an Image object or list of Image objects to plot
    - ax: a matplotlib axis to plot on (optional)
    - labels: a list of labels for the Image spectra (optional)
    - plot_blackbody: whether to plot a blackbody spectrum (optional)
    - temperature: the temperature to use for the blackbody spectrum (required if plot_blackbody is True)
    - plot_MC: whether to plot Monte Carlo spectra with error bars (optional)
    - MC_spec: a list of MCSpec objects to plot (required if plot_MC is True)
    - MC_labels: a list of labels for the MC spectra (optional)

    Output:
    - im: a list of matplotlib line objects for the Image spectra
    '''
    if not isinstance(image, list):
        image = [image]
    
    if labels is not None and not isinstance(labels, list):
        raise TypeError("labels must be a list of strings, not a single string")
    
    if labels is not None and len(labels) != len(image):
        raise ValueError(f"labels must have same length as image list ({len(labels)} != {len(image)})")
    
    #establish Hz as the normal frequency baseline and just apply a frequency_unit throughout
    if freq_units == 'eV':
        frequency_unit = Units.h_ev
        ax.set_xlabel('eV')
    else:
        frequency_unit = 1
        ax.set_xlabel('Hz')
    #establish erg as the normal energy baseline and just apply a lum_unit throughout
    if lum_units == 'eV':
        lum_unit = 1/Units.eV_2_erg
        ax.set_label('eV')
    else:
        lum_unit =1
        ax.set_label('erg')

    for i in range(len(image)):
        frequencies = image[i].frequencies
        L = image[i].get_luminosity()
        if (ax is None) and i==0:
             ax = plt.gca()

            
        ax.plot(frequencies*frequency_unit, L*frequencies*lum_unit, label=labels[i] if labels is not None else 'Blacklight Spectrum {0}'.format(i),marker='.',markersize=4)
        
    
    if plot_blackbody:
        if temperature is None:
            temperature = fit_blackbody(frequencies,L)
        bb_freq = np.logspace(np.log10(min(frequencies)), np.log10(max(frequencies)), 100)
        bb_flux = blackbody(bb_freq, temperature)
        ax.plot(bb_freq*frequency_unit, bb_flux*bb_freq*lum_unit*area, label='Blackbody (T={0:.2e} K)'.format(temperature))
    

    if MC_spec ==None and len(MC_spec_args)==5:
        MC_spec = MCSpec(MC_spec_args['directory'],MC_spec_args['nproc'],nfreq=MC_spec_args['nfreq'],emin=MC_spec_args['emin'],emax=MC_spec_args['emax'])
    elif MC_spec ==None and len(MC_spec_args)==2:
            MC_spec = MCSpec(MC_spec_args['directory'],MC_spec_args['nproc'])
    if MC_spec != None:
        if not isinstance(MC_spec, list):
            MC_spec = [MC_spec]

        for i in range(len(MC_spec)):
            ax.errorbar(MC_spec[i].freq*frequency_unit, MC_spec[i].lum*lum_unit,yerr=MC_spec[i].lum_err, label=MC_labels[i] if MC_labels is not None else 'MC Spectrum {0}'.format(i),marker='s',markersize=4)
            
            
    ax.legend()
    ax.set_xlabel('$\\nu$ ['+freq_units+']')
    ax.set_ylabel('$\\nu L_\\nu$ ['+lum_units+'$/s$]')
    return ax

def plot_manyIncs_spectra(blacklight_path,base_input_file,base_output_name,ninc, ax=None, plot_blackbody=False, temperature=None, MC_spec=None, MC_spec_args={}, MC_labels=None,freq_units='eV',lum_units='erg'):
    outfiles = run_multiple_jobs.run_mult_inclinations(blacklight_path,base_input_file,base_output_name,ninc)
    labels = ["i= "+str(o[-9:-3]) for o in outfiles]
    images = [Image(o) for o in outfiles]
    plot_spectra(images, ax=ax, labels=labels, plot_blackbody=plot_blackbody, temperature=temperature, MC_spec=MC_spec, MC_spec_args=MC_spec_args, MC_labels=MC_labels,freq_units=freq_units,lum_units=lum_units)

def plot_manytimes_spectra(outfiles, outputFig, plot_blackbody=False, temperature=None, MC_spec=None, MC_spec_args={}, MC_labels=None,freq_units='eV',lum_units='erg'):
    labels = ["t= "+str(o[-7:-3]) for o in outfiles]
    images = [Image(o) for o in outfiles]
    fig,ax = plt.subplots(1,1,figsize=(12,8))

    plot_spectra(images, ax=ax, labels=labels, plot_blackbody=False, temperature=None, MC_spec=None, MC_spec_args={}, MC_labels=None,freq_units='eV',lum_units='erg')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(outputFig)

def plot_image(image,image_name,freq=0,ax=None,axes='rg',logc=False,cmin=None,cmax=None,level=0):
    
    if axes == 'rg' or (axes is None and image.mass_msun is None):
        half_width = 0.5 * image.width_rg
        scale_exponent = int('{0:24.16e}'.format(half_width).split('e')[1])
        if scale_exponent in (0, 1):
            scale = 1.0
            x_label = r'$x$ ($GM/c^2$)'
            y_label = r'$y$ ($GM/c^2$)'
        else:
            scale = 10.0 ** scale_exponent
            x_label = r'$x$ ($10^{' + repr(scale_exponent) + r'}\ GM/c^2$)'
            y_label = r'$y$ ($10^{' + repr(scale_exponent) + r'}\ GM/c^2$)'
        half_width /= scale
        extent = np.array((-half_width, half_width, -half_width, half_width))

    # Calculate root grid in cm
    elif axes == 'cm':
        rg = Units.gg_msun * image.mass_msun / Units.c ** 2
        half_width = 0.5 * image.width_rg * rg
        scale_exponent = int('{0:24.16e}'.format(half_width).split('e')[1])
        if scale_exponent in (0, 1):
            scale = 1.0
            x_label = r'$x$ ($\mathrm{cm}$)'
            y_label = r'$y$ ($\mathrm{cm}$)'
        else:
            scale = 10.0 ** scale_exponent
            x_label = r'$x$ ($10^{' + repr(scale_exponent) + r'}\ \mathrm{cm}$)'
            y_label = r'$y$ ($10^{' + repr(scale_exponent) + r'}\ \mathrm{cm}$)'
        half_width /= scale
        extent = np.array((-half_width, half_width, -half_width, half_width))

    # Calculate root grid in muas
    else:
        rg = Units.gg_msun * image.mass_msun / Units.c ** 2
        half_width = np.arctan(0.5 * image.width_rg / (image.distance)) / Units.muas
        scale_exponent = int('{0:24.16e}'.format(half_width).split('e')[1])
        if scale_exponent in (0, 1):
            scale = 1.0
            x_label = r'$x$ ($\mathrm{\mu as}$)'
            y_label = r'$y$ ($\mathrm{\mu as}$)'
        else:
            scale = 10.0 ** scale_exponent
            x_label = r'$x$ ($10^{' + repr(scale_exponent) + r'}\ \mathrm{\mu as}$)'
            y_label = r'$y$ ($10^{' + repr(scale_exponent) + r'}\ \mathrm{\mu as}$)'
        half_width /= scale
        extent = np.array((-half_width, half_width, -half_width, half_width))
    
    if image_name=='I':
        pic = image.get_I(level)
    elif image_name=='Q':
        pic = image.get_Q(level)
    elif image_name=='U':
        pic = image.get_U(level)
    elif image_name=='V':
        pic = image.get_V(level)
    else:
        pic = {0: image.get_Alternate_Image(image_name, 0)}
        if level > 0:
            raise UserWarning('The ability to plot adaptive-resolution images is still being developed!')
            for l in range(1, level + 1):
                pic[l] = image.get_Alternate_Image(image_name, l)

    zeroPic = np.array(pic[0], dtype=np.float64)
    if zeroPic.ndim == 3:
        zeroPic = zeroPic[freq, :, :]

    if level>0:
        raise UserWarning('The ability to plot adaptive-resolution images is still being developed!')
        zeroPic = image.get_I(0)
        if(np.array(zeroPic[0]).ndim==3):
            zeroPic = np.array(zeroPic[0],dtype=np.float64)[freq,:,:]
            for l in range(1,level+1):
                pic[l] = pic[l][freq,...]
        num_blocks_root_linear = zeroPic.shape[-1] / pic[1].shape[-1]
        block_width = (extent[1] - extent[0]) / num_blocks_root_linear
        extent_adaptive = {}
        for l in range(1, level + 1):
            block_width_level = block_width / 2 ** l
            extent_adaptive[l] = np.empty((image.adaptive_num_blocks[l], 4))
            for block in range(image.adaptive_num_blocks[l]):
                x_loc = image.block_locs[l][block,1]
                y_loc = image.block_locs[l][block,0]
                extent_adaptive[l][block,0] = extent[0] + x_loc * block_width_level
                extent_adaptive[l][block,1] = extent[0] + (x_loc + 1) * block_width_level
                extent_adaptive[l][block,2] = extent[2] + y_loc * block_width_level
                extent_adaptive[l][block,3] = extent[2] + (y_loc + 1) * block_width_level
    
        
    if ax is None:
        ax = plt.gca()
    
    if logc:
        colorpic =ax.imshow(zeroPic,extent=extent,norm=colors.LogNorm(vmin=cmin,vmax=cmax),origin='lower')
    else:
        colorpic =ax.imshow(zeroPic,extent=extent,origin='lower')

    # Plot adaptive image
    for l in range(1, level + 1):
        raise UserWarning('The ability to plot adaptive-resolution images is still being developed!')
        for block in range(image.adaptive_num_blocks[l]):
            if logc:
                print(np.argwhere(np.nonzero(pic[l][block,...])))
                colorpic = ax.imshow(pic[l][block,...], aspect='equal',
                    origin='lower', extent=extent_adaptive[l][block,:],norm=colors.LogNorm(vmin=cmin,vmax=cmax))
            else:
                #print(extent_adaptive[l][block,:])
                colorpic = ax.imshow(pic[l][block,...], aspect='equal',
                    origin='lower', extent=extent_adaptive[l][block,:])

        
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    #TO DO: add implementation to include proper title
    ax.set_title("Intensity at "+str(image.frequencies[freq]*Units.h_ev).format("0.2e")+" eV")
    return colorpic


def colorbar(ax: Axes, im: AxesImage):
    """
    Add a color bar aligned to `im` neater than `fig.colorbar(im)`.
    https://stackoverflow.com/a/39938019/8954109
    """

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    return ax.figure.colorbar(im, cax=cax, orientation="vertical")

def plot_2Dgeodesic_positions(geodesics,i,j,ax=None,stepNum=5):
    '''Plot 2D geodesic paths as x1 vs x2'''
    geo_x1 = geodesics.data['sample_pos'][i+int(np.sqrt(geodesics.npix))*j,::stepNum,1]
    geo_x2 = geodesics.data['sample_pos'][i+int(np.sqrt(geodesics.npix))*j,::stepNum,2]
    if ax is None:
        ax = plt.gca()
    ax.scatter(geo_x1,geo_x2,c=range(np.shape(geo_x1)[0]))
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    return ax

def plot_3Dgeodesic_positions(geodesics,i,j,coords='cart',stepNum=5):
    '''Plot interactive 3D geodesic paths in native simulation coordinates.'''

    import k3d
    from k3d.colormaps import matplotlib_color_maps
    geo_x1 = geodesics.data['sample_pos'][i+int(np.sqrt(geodesics.npix))*j,::stepNum,1]
    geo_x2 = geodesics.data['sample_pos'][i+int(np.sqrt(geodesics.npix))*j,::stepNum,2]
    geo_x3 = geodesics.data['sample_pos'][i+int(np.sqrt(geodesics.npix))*j,::stepNum,3]
    if coords == 'spherical':
        #convert from spherical to cartesian 
        temp_geo_x1 = geo_x1*np.sin(geo_x2)*np.cos(geo_x3)
        temp_geo_x2 = geo_x1*np.sin(geo_x2)*np.sin(geo_x3)
        temp_geo_x3 = geo_x1*np.cos(geo_x2)
        geo_x1 = temp_geo_x1
        geo_x2 = temp_geo_x2
        geo_x3 = temp_geo_x3
    
    vertices = np.vstack([geo_x1,geo_x2,geo_x3]).T
    plt_line = k3d.points(vertices,color_map=matplotlib_color_maps.Coolwarm,attribute=range(np.shape(geo_x1)[0]))
    plot = k3d.plot()
    plot += plt_line
    return plot

