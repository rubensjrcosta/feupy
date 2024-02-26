#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


from feupy.plotters.config import *
import matplotlib.pyplot as plt # A collection of command style functions
from gammapy.utils.scripts import make_path
from astropy import units as u

plt.style.use(make_path(PATHMYSTYLE))


# In[ ]:


__all__ = [
    "show_hist_counts",
    "show_sensitivity_curve",
]


# In[ ]:





# In[ ]:


def show_hist_counts(table, path_file=None):   
    fix, axes = plt.subplots(1, 4, figsize=(12, 4))
    axes[0].hist(table["counts"])
    axes[0].set_xlabel("Counts")
    axes[0].set_ylabel("Frequency");

    axes[1].hist(table["counts_off"])
    axes[1].set_xlabel("Counts Off");

    axes[2].hist(table["excess"])
    axes[2].set_xlabel("excess");

    axes[3].hist(table["sqrt_ts"])
    axes[3].set_xlabel(r"significance ($\sigma$)");
#     path_file =  utl.get_path_counts(region_of_interest)  
#     file_name = utl.name_to_txt(file_name)
    
#     savefig(path_file, file_name)
    return


# In[ ]:


def show_sensitivity_curve(table, path_file=None):
    # Plot the sensitivity curve
    
    is_s = table["criterion"] == "significance"

    fig, ax = plt.subplots()
    ax.plot(
        table["e_ref"][is_s],
        table["e2dnde"][is_s],
        "s-",
        color="red",
        label="significance",
    )

    is_g = table["criterion"] == "gamma"
    ax.plot(table["e_ref"][is_g], table["e2dnde"][is_g], "*-", color="blue", label="gamma")
    is_bkg_syst = table["criterion"] == "bkg"
    ax.plot(
        table["e_ref"][is_bkg_syst],
        table["e2dnde"][is_bkg_syst],
        "v-",
        color="green",
        label="bkg syst",
    )

    ax.loglog()
    # ax.set_xlabel(f"Energy [{table['e_ref'].unit.to_string(UNIT_STRING_FORMAT)}]")
    # ax.set_ylabel(f"Sensitivity [{table['e2dnde'].unit.to_string(UNIT_STRING_FORMAT)}]")
    ax.legend()
    plt.show()
    return


# In[ ]:


# from gammapy.maps.axes import UNIT_STRING_FORMAT
# # Plot the sensitivity curve
# t = sensitivity_table

# is_s = t["criterion"] == "significance"

# fig, ax = plt.subplots()
# ax.plot(
#     t["e_ref"][is_s],
#     t["e2dnde"][is_s],
#     "s-",
#     color="red",
#     label="significance",
# )

# is_g = t["criterion"] == "gamma"
# ax.plot(t["e_ref"][is_g], t["e2dnde"][is_g], "*-", color="blue", label="gamma")
# is_bkg_syst = t["criterion"] == "bkg"
# ax.plot(
#     t["e_ref"][is_bkg_syst],
#     t["e2dnde"][is_bkg_syst],
#     "v-",
#     color="green",
#     label="bkg syst",
# )

# ax.loglog()
# ax.set_xlabel(f"Energy [{t['e_ref'].unit.to_string(UNIT_STRING_FORMAT)}]")
# ax.set_ylabel(f"Sensitivity [{t['e2dnde'].unit.to_string(UNIT_STRING_FORMAT)}]")
# ax.legend()
# plt.show()

