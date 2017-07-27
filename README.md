chxanalys--CHX XPCS Data Analysis Packages
========

Repository for data collection and analysis scripts that are useful at the
CHX beamline at NSLS-II (11-id).

Installation instructions on Linux:

Install miniconda from https://conda.io/miniconda.html. Then create the environment for chxanalys:

conda create --name chxanalys python=3.6 numpy scipy matplotlib
source activate databroker
pip install -r https://raw.githubusercontent.com/yugangzhang/chxanalys/master/requirements.txt
pip install git+https://github.com/yugangzhang/chxanalys
