import polars as pl
import matplotlib.pyplot as plt
import numpy as np
basepath = "/run/media/kevivois/T7/BACHELOR/"
filepath = basepath + "aggregated.parquet"

points_per_sec = 40000

aggregated = pl.scan_parquet(filepath)
size = len(aggregated.collect())

cols = ["AccX", "AccY", "AccZ", "Sound"]
cnt = 0
def process(row):
    sensor_file = row["sensor_file"]

    sensor_df = pl.scan_parquet(basepath + "/" + sensor_file)

    summary = (
        sensor_df
        .with_row_index("idx")
        .with_columns((pl.col("idx") // points_per_sec).alias("group"))
        .group_by("group")
        .agg([
            pl.col(cols).mean().name.suffix("_mean"),
            pl.col(cols).min().name.suffix("_min"),
            pl.col(cols).max().name.suffix("_max"),
            pl.col(cols).std().name.suffix("_std"),
            pl.col(cols).median().name.suffix("_median"),
        ])
        .with_columns(pl.lit(sensor_file).alias("sensor_file"))
    )

    meta = pl.DataFrame([row]).lazy()

    joined  = summary.join(other=meta, on="sensor_file", how="left")
    global cnt
    cnt+=1
    print(f"{cnt}/{size} | {round(cnt/size,3)*100}%")
    return joined


'''
rows = aggregated.collect().to_dicts()

result = pl.concat([process(r).collect() for r in rows])

result = result.lazy()'''

result = pl.scan_csv("test.csv")

data = result.filter(pl.col("ToolIdx") == 1).with_row_index("x").collect()


fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

axes[0].scatter(
    data["x"],
    data["Sound_mean"],
    c=data["PassNumber"] % 10,
    cmap="tab10",
    s=5
)

axes[1].scatter(
    data["x"],
    data["Sound_min"],
    c=data["PassNumber"] % 10,
    cmap="tab10",
    s=5
)


axes[2].scatter(
    data["x"],
    data["Sound_max"],
    c=data["PassNumber"] % 10,
    cmap="tab10",
    s=5
)


axes[0].set_ylim(-0.02, 0.02)
axes[1].set_ylim(-5.0,1.0)
axes[2].set_ylim(0, 5.0)


data = data.with_columns(
    (pl.col("PassNumber") != pl.col("PassNumber").shift()).alias("change")
)
''
change_points = data.filter(pl.col("change"))["x"].to_list()
for x in change_points:
        axes[0].axvline(x, color="black", alpha=0.3, linewidth=1)
    
'''    
for row in data.iter_rows(named=True):
    if row["change"]:
        for i in range(3):
            axes[i].text(
                row["x"],
                row["Sound_mean"],
                str(row["PassNumber"]),
                fontsize=8,
                weight="bold"
            )

'''


plt.tight_layout()
plt.show()

'''
result.write_csv("test.csv")

print(result)

'''
