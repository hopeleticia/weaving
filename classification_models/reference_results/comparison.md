# Classifier Comparison

Source run IDs and all timing fields are in comparison.csv. 

| model | split | f1_percent | precision_percent | recall_percent | false_positive_rate_percent | training_seconds | test_prediction_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| catboost | validation | 84.535313 | 91.674208 | 78.427937 | 0.013129 | 156.794176 | 0.597287 |
| catboost | test | 84.016032 | 99.627431 | 72.634385 | 0.005564 | 156.794176 | 0.597287 |
| xgboost | validation | 84.269614 | 91.222407 | 78.301618 | 0.013888 | 57.995815 | 6.558636 |
| xgboost | test | 80.379507 | 99.617124 | 67.369423 | 0.005304 | 57.995815 | 6.558636 |
| lightgbm | validation | 84.259064 | 87.173790 | 81.532945 | 0.022112 | 26.717116 | 23.242896 |
| lightgbm | test | 91.707674 | 99.430508 | 85.098052 | 0.009984 | 26.717116 | 23.242896 |
