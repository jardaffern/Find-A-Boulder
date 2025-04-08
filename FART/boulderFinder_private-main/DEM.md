# Finding Awesome boulders RemoTely (FART)

## Modeling pipeline
these are in order of how they should be ran

NOTE: technically you would need to run `web_scraping.py`.

1. `dem_scraper.py`: This downloads TIF files and saves them.
2. `create_hillshades.py`: does a transformation of tif files to create hillshade
3. `split_tif_files.py`: Tile the .tif files in 300x300 chunks
4. `boulder_hillshade_labeler.py`: Loop through the tiled hillshades and intersect with the boulders. Creates the bbox for the various boulders if they are within a DEM
5. `setup_dem_training.py`: Parse out the relevant data from the hillshade labeler
6. `yolo_formatter.py`: Make .png files and do some formatting to get into COCO/YOLO format
7. `setup_train_valid.py`: Split up the train;validation datasets. Also remove DEMs that clearly failed
8. `yolo_object_det.py`: Short script that just does some sys commands for YOLO
8. `data.yaml`: This is the script needed to get arguments set up for YOLOV8

## One off scripts

1. `get_vedauwoo_spatial.py`: This converts a shapefile that JAL made into a dataframe object thatn be used for training data.
2. `read_manual_gps_boulders.py`: This script was some data that JAL entered into a csv manually and wanted to convert into a training data set.



## Next steps

This actually seemed to get the model to detect things:
- [x] rerun boudler hillshade labeler with larger zones. boudlers are way too small -> tried 6x6 for the buffer

- [ ] try rerunning on the 'high quality' dataset. This is: vedauwoo, hand picked, etc.

- [ ] Refactor code so there's some actual useful information that comes with the data (e.g. id)
- [ ] Try some data transformations. unsure if appropriate in this context


