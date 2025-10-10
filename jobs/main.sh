# Final inspection goes here
# It will run the scanner and save the images in ./data/raw: scan.py (to be written in the scripts folder using code from the scanner folder)
# It will crop the segments and save them in ./data/processed: crop.py (to be written in the scripts folder using code from the cropper folder)
# It will run the inspection using the model and pixel-wise comparison: inspect.py (already written in scripts using code from the src folder)
# It will send the results to the UI via an object/API calls: the UI will be coded in the web folder
# The frontend will be coded in the web folder using React and will display the results and allow the user to interact with them
# Specifically, it will allow a "Run Inspection" button that start the process described in this file.
# For one hexaboard, it will be scanned into multiple segments, each segment will be cropped and inspected (if it's not skipped).
# The results will be displayed in the UI where the user can move from one segment to another and see whether they are flagged by the model or by pixel-wise comparison (can be both).
# Then, they can inspect the segment in detail and mark it as "Good" or "Damaged" (this segment will be saved in the JSON damaged segment mapping like the 03 notebook in the notebooks folder).
