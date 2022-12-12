failed_jobs = """
    SELECT 
        mese, queue, n, fail, (fail * 100 / n)::numeric(15,3) perc
    FROM  (
        SELECT 
            substr(to_timestamp(eventtimeepoch)::text,1,7) mese,
            queue, 
            count(*) n, 
            SUM((jobstatus != 4 OR exitstatus != 0)::int) fail
        FROM htjob 
        WHERE
            eventtimeepoch BETWEEN to_unixtime(%s) AND to_unixtime(%s) AND
            runtime > 3600
        GROUP BY mese, queue 
        ORDER BY mese, n desc
    ) A;
"""

type_of_error_by_queue = """
    SELECT 
        queue, 
        jobstatus, (exitstatus != 0)::int exitstatus, 
        count(*) n,
        sum(runtime) sum_rt
    FROM htjob
    WHERE
        eventtimeepoch BETWEEN to_unixtime(%s) AND to_unixtime(%s) AND
        runtime > 3600
    GROUP BY queue, jobstatus, exitstatus;
"""

njobs_and_rt_perhour = """
    SELECT
        width_bucket(runtime/3600.0, 1, 48, 48) as hours,
        count(*) n,
        sum((jobstatus != 4 OR exitstatus != 0)::int) fail,
        (sum((jobstatus != 4 OR exitstatus != 0)::int) / sum(sum((jobstatus != 4 OR exitstatus != 0)::int)) over () * 100)::NUMERIC(15,3) as pct_fail,
        sum(runtime) sum_rt,
        (sum(runtime) / sum(sum(runtime)) over () * 100)::NUMERIC(15,3) as pct_rt
    FROM htjob
    WHERE 
        eventtimeepoch BETWEEN to_unixtime(%s) AND to_unixtime(%s)
    GROUP BY hours
    ORDER BY hours
"""

njobs_first_hour = """
    SELECT 
        width_bucket(runtime/300.0, 1, 12, 12) as five_minutes,
        count (*) n,
        sum((jobstatus != 4 OR exitstatus != 0)::int) fail
    FROM
        htjob
    WHERE
        runtime < 3600 and
        eventtimeepoch BETWEEN to_unixtime(%s) AND to_unixtime(%s)
    GROUP BY five_minutes
    ORDER BY five_minutes
"""

njobs_and_rt_perhour_and_queue = """
    SELECT
        width_bucket(runtime/3600.0, 1, 48, 48) as hours,
        queue,
        count(*) n
    FROM htjob
    WHERE 
        eventtimeepoch BETWEEN to_unixtime(%s) AND to_unixtime(%s) AND
        queue = ANY (%s) AND
        runtime > 3600
    GROUP BY hours, queue
    ORDER BY hours
"""

daily_job_submission_rate_and_slowdown = """
    WITH htjob_sept_dec AS (
        SELECT *
        FROM htjob
        WHERE 
            submittimeepoch BETWEEN to_unixtime(%s) AND to_unixtime(%s) 
    )
    SELECT
        CASE
            WHEN a.day > 1 and a.day < 7 THEN 'business day'
            ELSE 'weekend day'
        END as day_type,
        a.hour,
        AVG(n) avg_njobs_submitted,
        AVG(avg_slowdown) avg_slowdown
    FROM (
            SELECT 
                EXTRACT(isodow FROM to_timestamp(submittimeepoch)) as day,
                EXTRACT(HOUR FROM to_timestamp(submittimeepoch)) as hour,
                COUNT(*) n
            FROM htjob_sept_dec
            GROUP BY day, hour
        ) a 
        join (
            SELECT
                EXTRACT(isodow FROM to_timestamp(submittimeepoch)) as day,
                EXTRACT(HOUR FROM to_timestamp(submittimeepoch)) as hour,
                AVG((starttimeepoch - submittimeepoch + runtime * 1.0)/runtime) as avg_slowdown
            FROM htjob_sept_dec
            WHERE 
                starttimeepoch >= submittimeepoch and
                runtime != 0
            GROUP BY day, hour
        ) b 
        on a.day = b.day and a.hour = b.hour
    GROUP BY day_type, a.hour
"""

jobname_of_jobs = """
    SELECT distinct(substr(jobname, 2, length(jobname) - 2)) as jobname, queue
    FROM htjob
    WHERE 
        eventtimeepoch BETWEEN to_unixtime(%s) AND to_unixtime(%s) AND
        runtime > 3600
"""

jobs_from_date_to_date = """WITH A AS (
    SELECT 
       j.jobid||'.'||j.idx||'_'||j.hn job,
       jd.queue,
       (jd.jobstatus != 4 OR jd.exitstatus != 0)::int fail,
       MIN(j.ts) mint,
       MAX(j.ts) maxt,
       ARRAY_AGG(j.rt ORDER BY j.rt ASC) xt,
       ARRAY_AGG(j.rss ORDER BY j.rt ASC) x_j_ram,
       ARRAY_AGG(j.swp ORDER BY j.rt ASC) x_j_swap,
       ARRAY_AGG(j.disk ORDER BY j.rt ASC) x_j_disk,
       ARRAY_AGG(j.cpu1_t ORDER BY j.rt ASC) x_m_cpu1_t,
       ARRAY_AGG(j.cpu2_t ORDER BY j.rt ASC) x_m_cpu2_t,
       ARRAY_AGG(j.ram_pct ORDER BY j.rt ASC) x_m_ram_pct,
       ARRAY_AGG(j.swap_pct ORDER BY j.rt ASC) x_m_swap_pct,
       ARRAY_AGG(j.totload_avg ORDER BY j.rt ASC) x_m_totload_avg,
       ARRAY_AGG(j.selfage ORDER BY j.rt ASC) x_m_selfage
    FROM (
        SELECT *
        FROM (
            SELECT 
                ts, jobid, idx, queue, hn, rt, rss, swp, disk
            FROM hj
            WHERE ts BETWEEN to_unixtime(%s) - %s AND to_unixtime(%s) - %s
        ) j LEFT JOIN LATERAL (
            SELECT 
                cpu1_t,
                cpu2_t,
                ((memavail * 100.0) / memtot)::NUMERIC(15,2) as ram_pct,
                ((memavail * 100.0) / memtot)::NUMERIC(15,2) as swap_pct,
                totload_avg, 
                selfage
            FROM hm m
            WHERE extract(epoch FROM m.ts) >= j.ts AND j.hn = m.hn
            ORDER BY extract(epoch FROM m.ts)
            LIMIT 1
        ) m ON TRUE
    ) j INNER JOIN htjob jd ON
        j.queue = jd.queue AND
        j.jobid = jd.jobid AND j.idx = jd.idx AND
        j.ts BETWEEN jd.starttimeepoch AND jd.eventtimeepoch
    WHERE
      jd.eventtimeepoch BETWEEN to_unixtime(%s) AND to_unixtime(%s) AND
      jd.runtime >= %s
    GROUP BY job, jd.queue, fail ORDER BY mint
)
SELECT 
    job,
    queue,
    fail,
    mint,
    maxt,
    xt[:40] t,
    x_j_ram[:40] j_ram, 
    x_j_swap[:40] j_swap, 
    x_j_disk[:40] j_disk,
    x_m_cpu1_t[:40] m_cpu1_t,
    x_m_cpu2_t[:40] m_cpu2_t,
    x_m_ram_pct[:40] m_ram_pct,
    x_m_swap_pct[:40] m_swap_pct,
    x_m_totload_avg[:40] m_totload_avg,
    x_m_selfage[:40] m_selfage
FROM A 
WHERE 
    xt[1] <= 180 AND 
    ARRAY_LENGTH(xt,1) >= 40
"""