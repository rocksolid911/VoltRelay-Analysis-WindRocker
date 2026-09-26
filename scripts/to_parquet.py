import pyarrow.csv as pc, pyarrow.parquet as pq, time, duckdb
D='Hackathon Data Set _ Gradient/'
for t in ['swap_events','station_hourly_status','riders','batteries','support_tickets','stations','city_daily_context','fleet_partners']:
    t0=time.time()
    tbl=pc.read_csv(D+t+'.csv', convert_options=pc.ConvertOptions(strings_can_be_null=True, timestamp_parsers=['%Y-%m-%d %H:%M:%S','%Y-%m-%d']))
    pq.write_table(tbl,'data/parquet/%s.parquet'%t, compression='zstd')
    print(t,tbl.num_rows,round(time.time()-t0,1),'s')
con=duckdb.connect()
for t in ['swap_events','station_hourly_status','support_tickets','riders','batteries','stations','city_daily_context']:
    print('==',t); print(con.execute(f"describe select * from 'data/parquet/{t}.parquet'").df().iloc[:,:2].T.to_string(header=False))
