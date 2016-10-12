COMPRESSING YOUR SUBMISSION:

Suppose you have two files to submit
1) run.sh
2) test.py

run the command `./compress.sh run.sh test.py`. This should create a file named "kaggle_assignment.tar.gz.b64" which you need to upload on moodle.

Sample "run.sh" and "test.py" are provided to you. Just compress and upload them on moodle for trial.



DECOMPRESSING YOUR SUBMISSION:

If you want to decompress your compressed submission, run the command `./decompress.sh kaggle_assignment.tar.gz.b64`.



NOTE:

Each submission must contain "run.sh" file which takes two arguments as `./run.sh <train_file_path> <test_file_path>` and generated "output.csv" file containing labels for test examples. Assume that formats of "train file" and "test file" are identical to the corresponding files on Kaggle. Format of "output.csv" file should be identical to the submission file format of kaggle.

Please refer to moodle for more details.
