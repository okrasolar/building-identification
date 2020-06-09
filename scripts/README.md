# Scripts

These scripts serve as the entrypoints to the pipeline.

#### [export.py](export.py)

Downloads data, locally for the [HRSL dataset](https://ciesin.columbia.edu/data/hrsl/), and to the linked
Google Drive account for the [Earth engine data](https://developers.google.com/earth-engine/exporting).

#### [process.py](process.py)

Processed the HRSL data, specifically by regridding it so that it is on the same grid as the Earth Engine Sentinel data.

#### [engineer.py](engineer.py)

Combines the HRSL and Earth Engine datasets, and prepares them so they can be ingested by the machine learning models.
Specifically, this means that the data is split into (by default) 224 x 224 pixel (x, y) pairs, where the x array is
the input Earth Engine data and the y array is the HRSL mask for the same area, regridded onto the Earth Engine Sentinel
grid.


#### [train_models.py](train_models.py)

Trains a model on the engineered data.
