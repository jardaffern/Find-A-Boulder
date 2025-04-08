Set-Location -Path C:\Users\jlomb\Documents\PersonalProjects\MPExtensions
conda activate

"Dem - scraping"
python .\scrapingData\dem_scraper.py

"Creating hill shades"
python .\generatingData\create_hillshades.py

"Splitting tif files"
python .\generatingData\split_tif_files.py

"Subsetting to boulders with TIF files"
python .\generatingData\boulder_hillshade_labeler.py

"subset data from boulders"
python .\generatingData\setup_dem_training.py

"Spit data into yolo format"
python .\modeling\yolo\yolo_formatter.py

"Split data into train val datasets"
python .\modeling\yolo\setup_train_valid.py

