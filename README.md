# anomaly-detection_failure-prediction-CNAF-INFN

## SCALETTA LAVORO
* data analysis
    * job error type rate by queue
    * job runtime by queue
    * job runtime
    * job failure by runtime
    * slowdown by job type ($\frac{\text{waittime}+ \text{runtime}}{\text{runtime}}$)
    * daily submission rate on business days or weekend
    * checks what happens when it fails
* preprocessing (<i>micro panel data = multiple multivariate time series</i>)
    * add features that represent the longitudinality:
        * <u>time</u>: bin time steps and treat each bin as a separate column, ignoring temporal order
        * <u>group</u>: add a column that indicates the membership to a group
        * array of classifier or regressor trained with feature sets of all the resource units
    * check to do for time series:
        * same amount of information
        * stationary
        * seasonality
* define a <u>cost model</u> with domain expert to optimize ML models instead of meaningless f1-accuracy-recall scores
* task di ML possibili
    * job failure prediction
    * job wall time prediction