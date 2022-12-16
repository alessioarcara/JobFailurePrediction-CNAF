import sys
import gzip
import pandas as pd
from joblib import load


COLUMN_NAMES = [
    "job_id", "group", "job_type", "host_name",
    "just_started", "just_finished", "job_status",
    "cores_used", "power_HS6_core", "power_HS6_machine",
    "cpu_time", "job_time", "owner", "ram_memory",
    "total_memory", "unix_time", "fromhost", "disk"
]

CATEGORICAL_VARS = [*COLUMN_NAMES[1:6], "job_status", "owner"]

STRING_VARS = ['job_id', 'fromhost']


class classifier:
    def __init__(self, inputf):
        self.model = load(inputf)

    def predict():
        print("predicted")


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        print("Usage: python classifier.py <model dump> <log file>")
        exit(1)

    data = pd.read_csv(
        args[0],
        sep=" ",
        names=COLUMN_NAMES,
        dtype={c: "category" for c in CATEGORICAL_VARS},
        compression="gzip"
    )

    for VAR in STRING_VARS:
        data[VAR] = data[VAR].astype("string")

    data.insert(0, 'job', data['job_id'] + "_" + data['fromhost'])
    data.drop(['job_id', 'fromhost'], axis=1, inplace=True)

    
   # j.jobid||'.'||j.idx||'_'||jd.fromhost job,
   # jd.queue,
   # (jd.jobstatus != 4 OR jd.exitstatus != 0)::int fail,
   # MIN(j.ts) mint,
   # MAX(j.ts) maxt,
   # ARRAY_AGG(j.rt ORDER BY j.rt ASC) xt,
   # ARRAY_AGG(j.rss ORDER BY j.rt ASC) x_j_ram,
   # ARRAY_AGG(j.swp ORDER BY j.rt ASC) x_j_swap,
   # ARRAY_AGG(j.disk ORDER BY j.rt ASC) x_j_disk,
   # ARRAY_AGG(j.cpu1_t ORDER BY j.rt ASC) x_m_cpu1_t,
   # ARRAY_AGG(j.cpu2_t ORDER BY j.rt ASC) x_m_cpu2_t,
   # ARRAY_AGG(j.ram_pct ORDER BY j.rt ASC) x_m_ram_pct,
   # ARRAY_AGG(j.swap_pct ORDER BY j.rt ASC) x_m_swap_pct,
   # ARRAY_AGG(j.totload_avg ORDER BY j.rt ASC) x_m_totload_avg,
   # ARRAY_AGG(j.selfage ORDER BY j.rt ASC) x_m_selfage
    print(data.info())
