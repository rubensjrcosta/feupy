#!/usr/bin/env python
# coding: utf-8

# In[1]:


from feupy.config import *


# In[ ]:


__all__ = [
    "show_sky_map",
]


# In[4]:


import matplotlib.pyplot as plt # A collection of command style functions
from gammapy.utils.scripts import make_path


FILE_PATH = "$PYTHONPATH/feupy/plotters/"
path_my_plot_style = f"{FILE_PATH}/my_plot_style.txt" 
plt.style.use(make_path(path_my_plot_style))


# In[1]:


from matplotlib.patches import Circle  # $matplotlib/patches.py
from astropy.coordinates import SkyCoord


from gammapy.maps import Map
from gammapy.utils.regions import SphericalCircleSkyRegion
# help(Map)

def show_sky_map(
    name, 
    leg_style, 
    sources, 
    datasets,
    pulsars,
    roi,
    width=(20 * u.deg, 20 * u.deg)):
    
    frame  = "icrs" # International Celestial Reference System (ICRS)
    unit   = "deg"  # Degrees units
    fontsize = 6.5
    rotation = 20
    s=30
    offset_text_ra = -.05
    offset_text_dec = .0
    horizontalalignment='left'
    verticalalignment='center_baseline' # 'top', 'bottom', 'center', 'baseline', 'center_baseline'
        
#     region = SphericalCircleSkyRegion(center, radius=5 * u.deg) # defines the region

    target_name = roi.name
    radius = roi.radius
    position = roi.position 
    pos_ra = position.ra.value
    pos_dec= position.dec.value
            
    center = SkyCoord(pos_ra, pos_dec, unit=unit, frame=frame) # Source Position
        
    fig, ax = plt.subplots(figsize=(7, 7))

    kwargs = {"ax": ax}
    offset_lim = 2

    xlim = [center.ra.value+offset_lim, center.ra.value-offset_lim]
    ylim = [center.dec.value+offset_lim, center.dec.value-offset_lim]
    
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    
    marker = "P"
    color = "b"
    
    circle = Circle((pos_ra,pos_dec), radius=radius.value, edgecolor="b", lw=.5, ls='--',facecolor='none')
    
    ax.add_artist(circle)
    ax.annotate(
        f"{target_name} ({radius})",
        (pos_ra, pos_dec),
        fontsize=fontsize,
        horizontalalignment=horizontalalignment, 
        verticalalignment=verticalalignment,
        color=color,
        xytext=(
            pos_ra+offset_text_ra+1.9,
            pos_dec-offset_text_dec-.7
        ),
        arrowprops = dict(arrowstyle="-", color=color)
    )

    ax.scatter(
        pos_ra,
        pos_dec,
        s=s+50,
        label=target_name, 
        marker=marker, 
        edgecolor=color, 
        ec='black', 
        lw=.5,
        facecolor=color)
            
            
    ax.yaxis.set_units(u.Unit("degree"))
    ax.xaxis.set_units(u.Unit("degree"))
    
    for index, source in enumerate(sources):
                    
            source_pos = source.position
            pos_ra = source_pos.ra.value
            pos_dec= source_pos.dec.value
            source_name = datasets[index].name
            
            color = leg_style[source_name][0]
            marker = leg_style[source_name][1]

            ax.scatter(
                pos_ra,
                pos_dec,
                s=s+50,
                label=source_name, 
                marker=marker, 
                edgecolor=color, 
                ec='black', 
                lw=.5,
                facecolor=color
            )
            
    for index, source in enumerate(pulsars):

        source_pos = source.position
        pos_ra = source_pos.ra.value
        pos_dec= source_pos.dec.value
        source_name = source.name

        color = leg_style[source_name][0]
        marker = leg_style[source_name][1]

        ax.scatter(
            pos_ra,
            pos_dec,
            s=550,
            label=source_name, 
            marker= "*", 
            edgecolor=None, 
            ec='black', 
            lw=.5,
            facecolor=None
        )
            
#     lines = plt.gca().get_lines()
#     include = [0,1]
#     legend1 = plt.legend([lines[i] for i in include],[lines[i].get_label() for i in include], loc=1)
#     legend2 = plt.legend([lines[i] for i in [2,3]],['manual label 3','manual label 4'], loc=4)
# plt.gca().add_artist(legend1)
# plt.show()

    ax.legend(labelcolor="black", markerscale=.75)
    
    sky_map = create_sky_map(center, width)
    sky_map.plot(ax)

    plt.xlabel(r'Righ Ascension (deg)')   
    plt.ylabel(r'Declination (deg)')
    ax.set_facecolor("white")
    
#     path_file =  utl.get_path_sky_maps(region_of_interest)  
#     file_name = utl.name_to_txt(name)
        
#     savefig(path_file, file_name)
    
    # plt.legend(*scatter.legend_elements())
    return


# In[ ]:


def create_sky_map(center, width=(10 * u.deg, 10 * u.deg)):
    frame  = "icrs" # International Celestial Reference System (ICRS)
    unit   = "deg"  # Degrees units
    
    return Map.create(
        width=width,
        skydir=center, # Coordinate of map center
        proj="CAR", # ??
        binsz=0.05 *u.deg, # Pixel size in degrees
        map_type="wcs", # {'wcs', 'wcs-sparse', 'hpx', 'hpx-sparse', 'region'}
        frame= frame # Galactic ("galactic") or Equatorial ("icrs")
    )

