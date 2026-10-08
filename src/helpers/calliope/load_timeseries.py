from pathlib import Path

import pandas as pd


def load_country_timeseries(
    country: str,
    start_date: str,
    end_date: str,
    demand_path: Path = Path("resources/timeseries/demand.parquet"),
    vres_dir: Path = Path("resources/timeseries/renewables_ninja"),
) -> pd.DataFrame:
    """Load one country's demand and VRES profiles over [start_date, end_date)."""

    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")

    if end <= start:
        raise ValueError("end_date must be later than start_date.")

    # ------------------------------------------------------------------
    # Demand
    # ------------------------------------------------------------------

    # Only load the timestamp and selected-country columns.
    demand = pd.read_parquet(
        demand_path,
        columns=["timestamp", country],
    )

    demand["timestamp"] = pd.to_datetime(
        demand["timestamp"],
        utc=True,
    )
    demand = demand.set_index("timestamp")

    # ------------------------------------------------------------------
    # VRES availability
    # ------------------------------------------------------------------

    vres = pd.read_parquet(
        vres_dir / f"{country}.parquet",
        columns=[
            "solar",
            "onshore_wind",
            "offshore_wind",
        ],
    )

    vres.index = pd.to_datetime(vres.index, utc=True)
    vres.index.name = "timesteps"

    # ------------------------------------------------------------------
    # Apply [start, end) horizon
    # ------------------------------------------------------------------

    demand = demand.loc[
        (demand.index >= start)
        & (demand.index < end)
    ]

    vres = vres.loc[
        (vres.index >= start)
        & (vres.index < end)
    ]

    # ------------------------------------------------------------------
    # Validate alignment
    # ------------------------------------------------------------------

    if not demand.index.equals(vres.index):
        missing_demand = vres.index.difference(demand.index)
        missing_vres = demand.index.difference(vres.index)

        raise ValueError(
            f"Demand/VRES timestamps do not align for {country} "
            f"between {start} and {end}. "
            f"Missing demand hours: {len(missing_demand)}; "
            f"missing VRES hours: {len(missing_vres)}."
        )

    # ------------------------------------------------------------------
    # Assemble model time series
    # ------------------------------------------------------------------

    timeseries = pd.DataFrame(
        {
            "demand_power": demand[country],
            "solar": vres["solar"],
            "onshore_wind": vres["onshore_wind"],
            "offshore_wind": vres["offshore_wind"],
        },
        index=vres.index,
    )

    timeseries.index.name = "timesteps"

    return timeseries

def build_calliope_timeseries(
    country: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:

    df = load_country_timeseries(
        country=country,
        start_date=start_date,
        end_date=end_date,
    )

    calliope_df = pd.DataFrame(
        {
            ("demand_power", "sink_use_equals"): df["demand_power"],
            ("solar", "source_use_max"): df["solar"],
            ("onshore_wind", "source_use_max"): df["onshore_wind"],
            ("offshore_wind", "source_use_max"): df["offshore_wind"],
        },
        index=df.index,
    )

    calliope_df.columns = pd.MultiIndex.from_tuples(
        calliope_df.columns,
        names=["techs", "inputs"],
    )

    return calliope_df