from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import databento as db
from dotenv import dotenv_values
from IPython.display import Markdown, display
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


TIMEZONE = "America/New_York"  # IANA examples: America/New_York, Europe/London, Asia/Shanghai, UTC


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    data_dir: Path
    api_key: str = field(repr=False)
    historical_gateway: str


@dataclass
class DownloadResult:
    run_dir: Path
    dbn_path: Path
    manifest_path: Path
    summary: dict
    preview: pd.DataFrame = field(repr=False)


def load_config(project_root=None):
    root = Path(project_root or Path(__file__).parent).resolve()
    env_path = root.parent / ".env"
    values = dotenv_values(env_path)
    names = (
        "DATABENTO_API_KEY",
        "DATABENTO_HISTORICAL_BASE_URL",
        "DATABENTO_TIMESERIES_ENDPOINT",
    )
    missing = [name for name in names if not values.get(name)]
    if missing:
        raise ValueError(f"Missing configuration in {env_path}: {', '.join(missing)}")
    base = values[names[1]].rstrip("/")
    gateway = base.removesuffix("/v0")
    parsed = urlsplit(gateway)
    if parsed.scheme != "https" or not parsed.netloc or parsed.path or parsed.query or parsed.fragment or parsed.username:
        raise ValueError("DATABENTO_HISTORICAL_BASE_URL must be an HTTPS origin, optionally followed by /v0")
    if values[names[2]].rstrip("/") != f"{gateway}/v0/timeseries.get_range":
        raise ValueError("DATABENTO_TIMESERIES_ENDPOINT must match the configured historical gateway and /v0/timeseries.get_range")
    return PipelineConfig(root, root / "data", values[names[0]], gateway)


def create_historical_client(config):
    return db.Historical(key=config.api_key, gateway=config.historical_gateway)


def _request_time(value):
    try:
        zone = ZoneInfo(TIMEZONE)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"TIMEZONE must be a valid IANA timezone: {TIMEZONE!r}") from exc
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("start and end must be valid timestamps, not NaT")
    if timestamp.tzinfo is None:
        try:
            timestamp = timestamp.tz_localize(zone, ambiguous="raise", nonexistent="raise")
        except ValueError as exc:
            raise ValueError(f"Ambiguous or nonexistent local time {value!r} in {TIMEZONE}; specify an explicit UTC offset or choose a valid local time. {exc}") from exc
    return timestamp.tz_convert("UTC")


def _display_preview(frame):
    result = frame.copy()
    if isinstance(result.index, pd.DatetimeIndex):
        result.index = result.index.tz_convert(TIMEZONE)
    for column, dtype in result.dtypes.items():
        if isinstance(dtype, pd.DatetimeTZDtype):
            result[column] = result[column].dt.tz_convert(TIMEZONE)
    return result


def _write_manifest(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _prepare_run(data_dir, request, preview_rows, chunk_rows, export_formats):
    if preview_rows < 0 or chunk_rows < 1:
        raise ValueError("preview_rows must be nonnegative and chunk_rows must be positive")
    formats = tuple(dict.fromkeys(export_formats))
    if set(formats) - {"csv", "parquet"}:
        raise ValueError("export_formats accepts only 'csv' and 'parquet'")
    now = datetime.now(timezone.utc)
    run_dir = Path(data_dir) / now.strftime("%Y-%m-%d") / f"{now:%H%M%S_%f}_{uuid4().hex[:6]}"
    run_dir.mkdir(parents=True)
    manifest = {
        "status": "started",
        "mode": "historical",
        "created_at": now.isoformat(),
        "request": dict(request),
        "preview_rows_limit": preview_rows,
        "chunk_rows": chunk_rows,
        "export_formats": formats,
    }
    _write_manifest(run_dir / "manifest.json", manifest)
    return run_dir, manifest, formats


def _summarize_frames(frames, preview_rows):
    pieces = []
    retained = 0
    record_count = 0
    time_start = None
    time_end = None
    fields = []
    time_column = None
    missing_values = {}
    partial_bars = 0
    for frame in frames:
        if not fields:
            fields = [{"name": frame.index.name, "dtype": str(frame.index.dtype)}]
            fields.extend({"name": name, "dtype": str(dtype)} for name, dtype in frame.dtypes.items())
            time_column = frame.index.name
        record_count += len(frame)
        for name, count in frame.isna().sum().items():
            missing_values[name] = missing_values.get(name, 0) + int(count)
        if "partial_from_request" in frame:
            partial_bars += int(frame["partial_from_request"].sum())
        if not frame.empty:
            first, last = frame.index.min(), frame.index.max()
            time_start = first if time_start is None else min(time_start, first)
            time_end = last if time_end is None else max(time_end, last)
        if retained < preview_rows:
            piece = frame.head(preview_rows - retained).copy()
            pieces.append(piece)
            retained += len(piece)
    preview = pd.concat(pieces) if pieces else pd.DataFrame()
    return {
        "record_count": record_count,
        "time_column": time_column,
        "time_start": None if time_start is None else time_start.isoformat(),
        "time_end": None if time_end is None else time_end.isoformat(),
        "fields": fields,
        "missing_values": {name: count for name, count in missing_values.items() if count},
        "preview_rows": len(preview),
        "partial_bars": partial_bars,
    }, preview


def _summarize_store(store, dbn_path, request, preview_rows, chunk_rows, export_formats):
    schema = str(request.get("schema", store.schema))
    summary, preview = _summarize_frames(store.to_df(schema=schema, count=chunk_rows), preview_rows)
    if summary["record_count"] == 0:
        summary, preview = _summarize_frames([store.to_df(schema=schema)], preview_rows)
    files = {dbn_path.name: dbn_path.stat().st_size}
    for extension in export_formats:
        path = dbn_path.parent / f"data.{extension}"
        if extension == "csv":
            store.to_csv(path, schema=schema, mode="x")
        elif summary["record_count"] == 0:
            store.to_df(schema=schema).to_parquet(path)
        else:
            store.to_parquet(path, schema=schema, mode="x")
        files[path.name] = path.stat().st_size
    summary.update(dataset=request.get("dataset"), schema=schema, source_schema=schema, output_schema=schema,
                   source_records=summary["record_count"], files=files, total_file_bytes=sum(files.values()))
    return summary, preview


def download_historical(client, request, data_dir, preview_rows=5000, chunk_rows=100000, export_formats=(), input_times=None):
    inspect.signature(client.timeseries.get_range).bind(**request, path=Path("data.dbn.zst"))
    run_dir, manifest, formats = _prepare_run(data_dir, request, preview_rows, chunk_rows, export_formats)
    dbn_path = run_dir / "data.dbn.zst"
    partial_path = run_dir / "data.dbn.zst.partial"
    manifest_path = run_dir / "manifest.json"
    manifest.update(timezone=TIMEZONE, input_times=input_times or {name: request.get(name) for name in ("start", "end")})
    _write_manifest(manifest_path, manifest)
    started = perf_counter()
    try:
        downloaded = client.timeseries.get_range(**request, path=partial_path)
        downloaded.reader.close()
        download_seconds = perf_counter() - started
        partial_path.rename(dbn_path)
        store = db.DBNStore.from_file(dbn_path)
        try:
            summary, preview = _summarize_store(store, dbn_path, request, preview_rows, chunk_rows, formats)
        finally:
            store.reader.close()
        summary.update(download_seconds=download_seconds, total_seconds=perf_counter() - started)
        manifest.update(status="complete", summary=summary)
        _write_manifest(manifest_path, manifest)
        return DownloadResult(run_dir, dbn_path, manifest_path, summary, preview)
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, elapsed_seconds=perf_counter() - started)
        _write_manifest(manifest_path, manifest)
        raise


def _resolve_schema(dataset, data_type, bar_interval):
    if dataset == "DBIX.CORE" and data_type in {"trades", "l1", "l2", "order_book"}:
        raise ValueError("DBIX.CORE provides index values, not executable trades or order books; use index, ohlcv, definition, or status")
    if data_type == "index":
        if dataset != "DBIX.CORE":
            raise ValueError(f"data_type='index' requires DBIX.CORE, not {dataset}")
        return "mbp-1", "mbp-1"
    if data_type == "ohlcv":
        if bar_interval not in {"1s", "1m", "5m", "15m", "30m", "1h", "1d"}:
            raise ValueError("bar_interval must be one of: 1s, 1m, 5m, 15m, 30m, 1h, 1d")
        source_interval = "1m" if bar_interval in {"5m", "15m", "30m"} else bar_interval
        return f"ohlcv-{source_interval}", f"ohlcv-{bar_interval}"
    schemas = {"trades": "trades", "l1": "cmbp-1" if dataset == "OPRA.PILLAR" else "mbp-1",
               "l2": "mbp-10", "order_book": "mbo", "definition": "definition", "status": "status"}
    if data_type not in schemas:
        raise ValueError("data_type must be one of: trades, ohlcv, l1, l2, order_book, definition, status, index")
    return schemas[data_type], schemas[data_type]


def _bar_window(request, interval):
    if interval not in {"5m", "15m", "30m"}:
        raise ValueError("Derived bar intervals must be 5m, 15m, or 30m")
    timestamps = []
    for name in ("start", "end"):
        value = pd.Timestamp(request[name])
        value = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        if value != value.floor("min"):
            raise ValueError("Derived bars require start and end aligned to whole UTC minutes")
        timestamps.append(value)
    if timestamps[0] >= timestamps[1]:
        raise ValueError("start must be earlier than end")
    return *timestamps, pd.Timedelta(minutes=int(interval[:-1]))


def _bar_frames(store, request, interval, chunk_rows):
    start, end, duration = _bar_window(request, interval)
    keys = ["ts_event", "publisher_id", "instrument_id"]
    aggregations = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "symbol": "first"}
    columns = ["publisher_id", "instrument_id", *aggregations]
    pending = None
    last_timestamp = None

    def finish(values):
        frame = values.set_index("ts_event")
        frame["symbol"] = frame["symbol"].astype("string")
        frame["partial_from_request"] = (frame.index < start) | (frame.index + duration > end)
        return frame

    for frame in store.to_df(schema="ohlcv-1m", count=chunk_rows):
        if frame.empty:
            continue
        frame = frame.sort_index(kind="stable")
        if last_timestamp is not None and frame.index[0] < last_timestamp:
            raise ValueError("Source OHLCV records must be ordered by ts_event across chunks")
        last_timestamp = frame.index[-1]
        values = frame[columns].reset_index()
        values["ts_event"] = values["ts_event"].dt.floor(duration)
        grouped = values.groupby(keys, as_index=False, sort=True, dropna=False).agg(aggregations)
        if pending is not None:
            grouped = pd.concat([pending, grouped], ignore_index=True).groupby(keys, as_index=False, sort=True, dropna=False).agg(aggregations)
        latest = grouped["ts_event"].max()
        complete = grouped[grouped["ts_event"] < latest]
        pending = grouped[grouped["ts_event"] == latest]
        if not complete.empty:
            yield finish(complete)
    if pending is not None:
        yield finish(pending)
    else:
        empty = store.to_df(schema="ohlcv-1m")[columns]
        empty.index = pd.DatetimeIndex([], dtype="datetime64[ns, UTC]", name="ts_event")
        yield finish(empty.reset_index())


def _aggregate_bars(result, request, interval, preview_rows, chunk_rows=100000, export_formats=()):
    _bar_window(request, interval)
    formats = tuple(dict.fromkeys(("parquet", *export_formats)))
    if set(formats) - {"csv", "parquet"}:
        raise ValueError("export_formats accepts only 'csv' and 'parquet'")
    paths = {extension: result.run_dir / f"bars_{interval}.{extension}" for extension in formats}
    partial_paths = {extension: path.with_suffix(path.suffix + ".partial") for extension, path in paths.items()}
    manifest = json.loads(result.manifest_path.read_text())
    manifest.update(status="processing", source_schema="ohlcv-1m", output_schema=f"ohlcv-{interval}",
                    export_formats=formats, preview_rows_limit=preview_rows, chunk_rows=chunk_rows,
                    aggregation={"interval": interval, "timezone": "UTC", "closed": "left", "label": "left",
                                 "group_by": ["publisher_id", "instrument_id"]})
    _write_manifest(result.manifest_path, manifest)
    started = perf_counter()
    store = db.DBNStore.from_file(result.dbn_path)

    def write_frames():
        writer = None
        first = True
        try:
            for frame in _bar_frames(store, request, interval, chunk_rows):
                table = pa.Table.from_pandas(frame, schema=None if writer is None else writer.schema, preserve_index=True)
                if writer is None:
                    writer = pq.ParquetWriter(partial_paths["parquet"], table.schema, compression="zstd")
                writer.write_table(table)
                if "csv" in formats:
                    frame.to_csv(partial_paths["csv"], mode="w" if first else "a", header=first)
                first = False
                yield frame
        finally:
            if writer is not None:
                writer.close()

    try:
        with closing(write_frames()) as frames:
            summary, preview = _summarize_frames(frames, preview_rows)
        for extension, path in paths.items():
            partial_paths[extension].rename(path)
        files = {result.dbn_path.name: result.dbn_path.stat().st_size}
        files.update({path.name: path.stat().st_size for path in paths.values()})
        summary.update(dataset=request["dataset"], schema=f"ohlcv-{interval}", source_schema="ohlcv-1m",
                       output_schema=f"ohlcv-{interval}", source_records=result.summary["record_count"],
                       bar_interval=interval, files=files, total_file_bytes=sum(files.values()),
                       download_seconds=result.summary["download_seconds"],
                       total_seconds=result.summary["total_seconds"] + perf_counter() - started)
        manifest.update(status="complete", summary=summary)
        _write_manifest(result.manifest_path, manifest)
        return DownloadResult(result.run_dir, result.dbn_path, result.manifest_path, summary, preview)
    except BaseException as exc:
        manifest.pop("summary", None)
        manifest.update(status="failed", error_type=type(exc).__name__)
        _write_manifest(result.manifest_path, manifest)
        raise
    finally:
        store.reader.close()


def download_and_show(*, dataset, symbols, start, end, data_type="trades", bar_interval="1m", stype_in="raw_symbol", export_formats=("parquet",), preview_rows=5000, data_dir=None):
    if preview_rows < 0 or set(export_formats) - {"csv", "parquet"}:
        raise ValueError("preview_rows must be nonnegative; export_formats accepts only 'csv' and 'parquet'")
    source_schema, output_schema = _resolve_schema(dataset, data_type, bar_interval)
    derived = source_schema != output_schema
    start_utc, end_utc = _request_time(start), _request_time(end)
    if start_utc >= end_utc:
        raise ValueError("start must be earlier than end")
    request = {"dataset": dataset, "symbols": symbols, "start": start_utc.isoformat(), "end": end_utc.isoformat(),
               "schema": source_schema, "stype_in": stype_in, "limit": None}
    if derived:
        _bar_window(request, bar_interval)
    config = load_config()
    client = create_historical_client(config)
    available = client.metadata.list_schemas(dataset=dataset)
    if data_type == "l1":
        source_schema = next((schema for schema in ("mbp-1", "cmbp-1") if schema in available), None)
        if source_schema is None:
            raise ValueError(f"{dataset} does not support data_type='l1'; neither mbp-1 nor cmbp-1 is available")
        output_schema = request["schema"] = source_schema
    if source_schema not in available:
        raise ValueError(f"{dataset} does not support data_type={data_type!r}, which requires {source_schema}")
    print(f"Downloading {dataset} / {output_schema}: {start_utc.tz_convert(TIMEZONE).isoformat()} to {end_utc.tz_convert(TIMEZONE).isoformat()} ({TIMEZONE})")
    result = download_historical(client, request, data_dir or config.data_dir,
                                       preview_rows=0 if derived else preview_rows,
                                       export_formats=() if derived else export_formats,
                                       input_times={"start": start, "end": end})
    if derived:
        result = _aggregate_bars(result, request, bar_interval, preview_rows, export_formats=export_formats)
    result.summary.update(data_type=data_type, source_schema=source_schema, output_schema=output_schema,
                          bar_interval=bar_interval if data_type == "ohlcv" else None)
    manifest = json.loads(result.manifest_path.read_text())
    manifest.update(selection={"data_type": data_type, "bar_interval": result.summary["bar_interval"]}, summary=result.summary)
    _write_manifest(result.manifest_path, manifest)
    show_download(result)
    return result


def show_download(result):
    summary = result.summary
    times = {name: None if summary[name] is None else pd.Timestamp(summary[name]).tz_convert(TIMEZONE).isoformat()
             for name in ("time_start", "time_end")}
    display(Markdown("**Download complete — summary of the full output**"))
    display(pd.Series({
        "Dataset": summary["dataset"],
        "Source schema": summary["source_schema"],
        "Output schema": summary["output_schema"],
        "Bar interval": summary.get("bar_interval"),
        "Downloaded source records": summary["source_records"],
        "Output bars" if summary["schema"].startswith("ohlcv") else "Output records": summary["record_count"],
        f"First timestamp ({TIMEZONE})": times["time_start"],
        f"Last timestamp ({TIMEZONE})": times["time_end"],
        "Timestamp field": summary["time_column"],
        "Download seconds": round(summary["download_seconds"], 2),
        "Total seconds": round(summary["total_seconds"], 2),
    }, name="Value").to_frame())
    print(f"Saved to: {result.run_dir}")
    display(pd.DataFrame([
        {"File": name, "Bytes": size, "MiB": round(size / 1024 ** 2, 3)}
        for name, size in summary["files"].items()
    ]))
    print(f"Request and results: {result.manifest_path.name}")
    print(f"Raw source DBN: {result.dbn_path.name}")
    if summary["partial_bars"]:
        print(f"{summary['partial_bars']:,} bars overlap a request boundary and are marked partial_from_request. Only data within the requested interval was used.")
    if summary.get("data_type") == "order_book":
        print("Order book output contains order events; book reconstruction is not performed.")
    missing = summary["missing_values"]
    if missing:
        display(Markdown("**Missing values across all output records — up to 10 fields**"))
        display(pd.DataFrame([
            {"Field": name, "Missing records": count, "Percent": round(count / summary["record_count"] * 100, 2)}
            for name, count in sorted(missing.items(), key=lambda item: item[1], reverse=True)[:10]
        ]))
    else:
        print("Missing values across all output records: 0.")
    display(Markdown(f"**Preview — first {len(result.preview):,} output records; first 10 shown below**"))
    display(_display_preview(result.preview.head(10)))
    numeric = [name for name in ("price", "bid_px_00", "ask_px_00", "close", "size", "volume") if name in result.preview]
    if summary["dataset"] == "DBIX.CORE":
        numeric = [name for name in numeric if name not in {"size", "volume"}]
    if numeric and not result.preview.empty:
        display(Markdown("**Numeric summary of the preview — pooled across instruments in the preview**"))
        display(result.preview[numeric].describe().T[["count", "mean", "min", "max"]])
    figure = plot_preview(result.preview, summary["schema"], dataset=summary["dataset"])
    display(figure)
    plt.close(figure)


def plot_preview(preview, schema, max_symbols=6, dataset=None):
    if max_symbols < 1:
        raise ValueError("max_symbols must be positive")
    if preview.empty:
        figure, axis = plt.subplots(figsize=(11, 4))
        axis.text(0.5, 0.5, "No records to display", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        return figure
    frame = preview.tz_convert("UTC").sort_index()
    group_column = "instrument_id" if "instrument_id" in frame else "symbol" if "symbol" in frame else None
    group_columns = ["publisher_id", "instrument_id"] if {"publisher_id", "instrument_id"}.issubset(frame.columns) else group_column
    multiple_publishers = {"publisher_id", "instrument_id"}.issubset(frame.columns) and frame["publisher_id"].nunique() > 1
    groups = []
    for identifier, group in frame.groupby(group_columns, sort=False, dropna=False) if group_columns else [("All records", frame)]:
        instrument = identifier[-1] if isinstance(identifier, tuple) else identifier
        symbols = group["symbol"].dropna().unique() if "symbol" in group else []
        label = f"{symbols[0]} ({instrument})" if len(symbols) == 1 and group_column == "instrument_id" else str(instrument)
        if multiple_publishers:
            label += f" · publisher {identifier[0]}"
        groups.append((label, group))
    plotted = groups[:max_symbols]
    ohlcv = str(schema).startswith("ohlcv")
    show_volume = ohlcv and dataset != "DBIX.CORE"
    figure, axes = plt.subplots(2 if show_volume else 1, 1, figsize=(11, 6 if show_volume else 4), sharex=True, squeeze=False)
    axis = axes[0, 0]
    quote_fields = [name for name in ("bid_px_00", "ask_px_00") if name in frame and frame[name].notna().any()]
    if dataset == "DBIX.CORE" and "price" in frame and frame["price"].notna().any():
        price_fields = ["price"]
    elif ohlcv:
        price_fields = ["close"]
    elif quote_fields:
        price_fields = quote_fields
    elif "price" in frame and frame["price"].notna().any() and str(schema) not in {"mbo", "statistics", "definition", "status", "imbalance"}:
        price_fields = ["price"]
    else:
        price_fields = []
    span = (frame.index.max() - frame.index.min()).total_seconds()
    bucket = pd.Timedelta(seconds=max(1, int(span / 60)))
    for label, group in plotted:
        if price_fields:
            for column in price_fields:
                axis.plot(group.index, group[column], label=f"{label} · {column}", linewidth=1, marker="." if len(group) == 1 else None)
        else:
            counts = pd.Series(1, index=group.index).resample(bucket).sum()
            axis.step(counts.index, counts, where="mid", label=str(label), marker="." if len(counts) == 1 else None)
        if show_volume:
            axes[1, 0].plot(group.index, group["volume"], label=str(label), linewidth=1, marker="." if len(group) == 1 else None)
    if not price_fields:
        axis.set_xlim(frame.index.min().floor(bucket) - bucket / 2, frame.index.max().ceil(bucket) + bucket / 2)
    if span == 0:
        seconds = {"ohlcv-1s": 1, "ohlcv-1m": 60, "ohlcv-5m": 300, "ohlcv-15m": 900, "ohlcv-30m": 1800, "ohlcv-1h": 3600, "ohlcv-1d": 86400, "ohlcv-eod": 86400}.get(str(schema), 30)
        window = pd.Timedelta(seconds=seconds)
        axis.set_xlim(frame.index[0] - window, frame.index[0] + window)
    zone = ZoneInfo(TIMEZONE)
    locator = mdates.AutoDateLocator(tz=zone)
    axes[-1, 0].xaxis.set_major_locator(locator)
    axes[-1, 0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=zone))
    axis.set_ylabel("Index value" if dataset == "DBIX.CORE" and price_fields else "Price" if price_fields else f"Events / {bucket}")
    axis.legend(fontsize=8)
    if show_volume:
        axes[1, 0].set_ylabel("Volume")
    axes[-1, 0].set_xlabel(f"Time ({TIMEZONE})")
    suffix = f" · first {max_symbols} of {len(groups)} symbols" if len(groups) > max_symbols else ""
    axis.set_title(f"{schema} · first {len(preview):,} output records{suffix}")
    for item in axes.flat:
        item.grid(alpha=0.2)
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure
