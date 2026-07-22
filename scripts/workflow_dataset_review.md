## Workflow of the dataset review

1. Extract the consumption label data from the downloaded json file 

1.1. Use the correct name for the json file in the extract_consumption_index.py script 
1.2. Go through that now readable json file compare for a view values whether it actually worked

2. Check for issues

2.1. Run the validate_data_consumed.py script. It list how many total erros you have and list the first ten errors.
2.2. Analyse if you have issues you are done. When you have 0 issues you are done and can skrip step 3. Otherwise continue with step 3.

3. Fix issues
3.1. Run the script review_consumption_webui.py it shows all files without issues green and with issues red. TODO: One could add a toggle.
Only the red tasks are important so those with issues. 
3.2. Click on the issue that is red. Read at the top the problem. Look at the image with its labels evaluate whether the flagged problem actually exists. Below you can manipulate the json file. If the manipulation isnt possible in this UI at the top you have the image id which you can then use to scan through your 
