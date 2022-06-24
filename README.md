# Introduction

Hi! Welcome to the ACES-Dashboard repository. 

This repo is one of three related to the Atlantic Canada Energy System (ACES) Model.
The others are:
- The [ACES-Temoa](https://github.com/SutubraResearch/ACES-Temoa) repository contains the code
for the energy system planning model, based off [Temoa](https://temoacloud.com/).
- The [ACES-Data](https://github.com/SutubraResearch/ACES-Data) repository contains the data files
associated with the ACES Model.

The dashboard is still in a beta version. Feel free to flag [issues](https://github.com/SutubraResearch/ACES-Dashboard/issues) and we'll 
do our best to make improvements and patches.
Otherwise, please feel free to get in touch @ cameron@sutubra.ca.

# Usage

## Getting the dashboard up-and-running
To use the dashboard, you will first need to activate the ACES environment, which you
would have installed from the [ACES-Temoa](https://github.com/SutubraResearch/ACES-Temoa) repository.

To fire up the dashboard on your machine, use the following command in a terminal / command prompt:
<pre><code>python Dashboard.py -i &ltinput filename&gt -p &ltplot type&gt [--super]
</code></pre>

where 
* `<input filename>` is the filename of the .sqlite database you wish to visualize.
*  `<plot type>` is one of:
    - `emissions` to generate the emissions dashboard
    - `capacity` to generate the dashboard for capacity and activity
    - `flow` to generate the dashboard for the hourly dispatch
* `--super` is an optional flag for the `capacity` plot types. This flag will result in 
similar technologies being grouped together and aggregated in the plots. 

Afer a moment, a url will appear. Simply paste this url into your web browser. Please visit
the [Dash website](https://dash.plotly.com/) for more details.

## Changing technology colours and other attributes
Information from the sqlite database (`<input filename>`) is used to generate certain plot features.

Specifically, the "technologies" table contains the following attributes used by the dashboard:
* the "sector" and "subsector" attributes are used to group technologies together for sorting.
* the "tech_desc" attribute is used in the plot legend.
* the "tech_category" attribute determines if and how technologies are aggregated when the 
`--super` flag is included in the capacity plots.
* the "cap_units" attribute is used as the y-axis unit, when relevant. 
* the "color" attribute is a HEX color used in the plots
* the "plot_order" attribute is an ordinal ranking where the technology should appear
in the stacked plots -- the higher the number, the higher in the stack the technology will be plotted.




  
  

