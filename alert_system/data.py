from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from uuid import uuid4

import databento as db
import pandas as pd
import pyarrow.parquet as pq

from .config import MAX_EVENTS, PROJECT_ROOT, SUPPORTED_DATASETS, SUPPORTED_SCHEMAS, TIMEZONE, create_historical_client, load_config


@dataclass
class MarketData:
    events: list[dict]
    metadata: dict


def utc_time(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("A valid timestamp is required")
    if stamp.tzinfo is None:
        try:
            stamp = stamp.tz_localize(TIMEZONE, ambiguous="raise", nonexistent="raise")
        except ValueError as exc:
            raise ValueError("Ambiguous or nonexistent local time; specify an explicit UTC offset") from exc
    return stamp.tz_convert("UTC")


def number(value, field, invalid_fields, allow_missing=False):
    if value is None or value is pd.NA:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        invalid_fields[field] = repr(value)
        return None
    if not math.isfinite(result):
        if not (allow_missing and math.isnan(result)):
            invalid_fields[field] = "NaN" if math.isnan(result) else "Infinity" if result > 0 else "-Infinity"
        return None
    return result


def _merge_context(metadata, declarations):
    for field in ("dataset", "schema", "asset_class"):
        value = declarations.get(field)
        if value is None or pd.isna(value):
            continue
        value = str(value)
        if metadata.get(field) is not None and metadata[field] != value:
            raise ValueError(f"Conflicting {field} declarations: {metadata[field]!r} and {value!r}")
        metadata[field] = value


def _validate_context(frame, metadata):
    metadata = dict(metadata)
    _merge_context(metadata, frame.attrs)
    for field in ("dataset", "schema", "asset_class"):
        if field not in frame:
            continue
        values = frame[field].dropna().astype(str).unique()
        if len(values) > 1:
            raise ValueError(f"Mixed {field} declarations are not supported in one input file")
        if len(values):
            _merge_context(metadata, {field: values[0]})
    if not metadata.get("dataset") or not metadata.get("schema"):
        raise ValueError("Dataset and schema context is required. Keep the Databento manifest.json beside CSV/Parquet files, or include consistent dataset and schema declarations in the file.")
    if metadata["dataset"] not in SUPPORTED_DATASETS or metadata.get("asset_class", "equity") != "equity":
        raise ValueError("This project supports equity datasets only")
    schema = metadata["schema"]
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"Unsupported schema {schema}; supported schemas: {', '.join(SUPPORTED_SCHEMAS)}")
    if "rtype" in frame:
        values = pd.to_numeric(frame["rtype"], errors="coerce")
        expected = 0 if schema == "trades" else 1
        if not values.eq(expected).all():
            raise ValueError(f"Record rtype conflicts with schema {schema}; expected rtype {expected}")
    if "depth" in frame and not pd.to_numeric(frame["depth"], errors="coerce").eq(0).all():
        raise ValueError("Only depth-zero trade or level-one quote events are supported")
    if "order_id" in frame or any(re.fullmatch(r"(?:bid|ask)_(?:px|sz|ct)_0*[1-9][0-9]*", str(field)) for field in frame):
        raise ValueError("Order-level or deeper-book fields conflict with the supported trades/mbp-1 schemas")
    if {"open", "high", "low", "close"}.issubset(frame):
        raise ValueError("OHLCV fields are not individual trade or level-one quote events")
    kind = "trade" if schema == "trades" else "quote"
    if "kind" in frame and not frame["kind"].dropna().eq(kind).all():
        raise ValueError(f"Record kind conflicts with schema {schema}")
    required = ("price", "size") if kind == "trade" else tuple(raw if raw in frame else normalized for raw, normalized in (("bid_px_00", "bid"), ("ask_px_00", "ask"), ("bid_sz_00", "bid_size"), ("ask_sz_00", "ask_size")))
    if any(field not in frame for field in required):
        raise ValueError(f"Schema {schema} requires fields: {', '.join(required)}")
    metadata["asset_class"] = "equity"
    return metadata


def supported_file(path):
    return path.name.endswith((".parquet", ".csv", ".dbn", ".dbn.zst"))


def local_path(relative, root=PROJECT_ROOT):
    base = (Path(root) / "data").resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base) or not path.is_file() or not supported_file(path):
        raise ValueError("Choose a supported Parquet, CSV, or DBN file inside the project data directory")
    return path


def list_files(root=PROJECT_ROOT):
    base = Path(root) / "data"
    if not base.exists():
        return []
    result = []
    for path in sorted(base.rglob("*")):
        if path.is_file() and supported_file(path) and path.resolve().is_relative_to(base.resolve()):
            result.append({"path": str(path.relative_to(base)), "label": str(path.relative_to(base)), "bytes": path.stat().st_size})
    return result


def validate_request(payload):
    source = payload.get("source", "bundled")
    if source not in {"bundled", "local", "databento"}:
        raise ValueError("source must be bundled, local, or databento")
    if source == "local":
        if not isinstance(payload.get("path"), str):
            raise ValueError("A local data file must be selected")
        return {"source": source, "path": payload["path"]}
    if source == "bundled":
        sample = payload.get("sample", "quotes")
        if sample not in {"trades", "quotes"}:
            raise ValueError("sample must be trades or quotes")
        return {"source": source, "sample": sample}
    dataset = payload.get("dataset", "XNAS.ITCH")
    schema = payload.get("schema", "mbp-1")
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError("Choose a supported equity dataset: " + ", ".join(SUPPORTED_DATASETS))
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError("schema must be trades or mbp-1")
    symbols = payload.get("symbols", [])
    if isinstance(symbols, str):
        symbols = [value.strip() for value in symbols.split(",") if value.strip()]
    if not isinstance(symbols, list) or not symbols or any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9./_\-]{1,32}", value) for value in symbols):
        raise ValueError("symbols must contain valid equity ticker symbols")
    start, end = utc_time(payload.get("start")), utc_time(payload.get("end"))
    if start >= end:
        raise ValueError("start must be earlier than end")
    if end > pd.Timestamp.now(tz="UTC"):
        raise ValueError("Historical requests require an end time in the past")
    return {"source": source, "dataset": dataset, "schema": schema, "symbols": list(dict.fromkeys(symbols)), "start": start.isoformat(), "end": end.isoformat(), "stype_in": "raw_symbol"}


def _bounded_store_frame(store):
    frames = []
    count = 0
    for frame in store.to_df(count=50000):
        count += len(frame)
        if count > MAX_EVENTS:
            raise ValueError(f"The detector accepts up to {MAX_EVENTS:,} events. Choose a shorter interval or split the file; downloaded source data is retained.")
        frames.append(frame)
    return pd.concat(frames) if frames else pd.DataFrame()


def _read_file(path, metadata):
    if path.name.endswith((".dbn", ".dbn.zst")):
        store = db.DBNStore.from_file(path)
        try:
            _merge_context(metadata, {"dataset": str(store.dataset), "schema": str(store.schema)})
            return _bounded_store_frame(store)
        finally:
            store.reader.close()
    if path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows > MAX_EVENTS:
            raise ValueError(f"The detector accepts up to {MAX_EVENTS:,} events. Split this file into smaller intervals.")
        headers = parquet.metadata.metadata or {}
        _merge_context(metadata, {field: headers[field.encode()].decode() for field in ("dataset", "schema", "asset_class") if field.encode() in headers})
        return pd.read_parquet(path)
    frame = pd.read_csv(path, nrows=MAX_EVENTS + 1)
    if len(frame) > MAX_EVENTS:
        raise ValueError(f"The detector accepts up to {MAX_EVENTS:,} events. Split this file into smaller intervals.")
    return frame


def normalize_frame(frame, metadata):
    if len(frame) > MAX_EVENTS:
        raise ValueError(f"The detector accepts up to {MAX_EVENTS:,} events. Choose a shorter interval.")
    if frame.empty:
        raise ValueError("The requested interval contains no market records")
    metadata = _validate_context(frame, metadata)
    if isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.reset_index(names=frame.index.name or "ts_event")
    timestamp_field = next((name for name in ("ts_event", "timestamp", "ts_recv") if name in frame), None)
    if timestamp_field is None:
        raise ValueError("Data must include a ts_event or timestamp column")
    if "symbol" not in frame:
        if "instrument_id" not in frame:
            raise ValueError("Data must include symbol or instrument_id")
        frame = frame.copy()
        frame["symbol"] = frame.instrument_id.map(lambda value: f"Instrument {value}")
    schema = metadata["schema"]
    kind = {"mbp-1": "quote", "trades": "trade"}[schema]
    fields = {"size": "size", "bid": "bid_px_00" if "bid_px_00" in frame else "bid", "ask": "ask_px_00" if "ask_px_00" in frame else "ask", "bid_size": "bid_sz_00" if "bid_sz_00" in frame else "bid_size", "ask_size": "ask_sz_00" if "ask_sz_00" in frame else "ask_size"}
    events = []
    for index, row in enumerate(frame.to_dict("records"), 1):
        timestamp = pd.Timestamp(row[timestamp_field])
        if pd.isna(timestamp):
            raise ValueError(f"Record {index} has an invalid timestamp")
        timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
        event = {"id": index, "timestamp": timestamp.isoformat(), "symbol": str(row["symbol"]), "dataset": metadata.get("dataset", "Imported Databento data"), "schema": schema, "publisher_id": None if pd.isna(row.get("publisher_id")) else int(row["publisher_id"]), "kind": kind, "asset_class": "equity"}
        event["source_record_id"] = index
        event["ts_event"] = timestamp.isoformat()
        for field in ("instrument_id", "sequence"):
            if row.get(field) is not None and not pd.isna(row[field]):
                event[field] = int(row[field])
        for field in ("action", "side"):
            if row.get(field) is not None and not pd.isna(row[field]):
                event[field] = str(row[field])
        invalid_fields = {}
        for name, original in fields.items():
            if original in row:
                size_field = fields.get(f"{name}_size")
                absent_side = name in ("bid", "ask") and (row.get(size_field) is None or pd.isna(row.get(size_field)) or row.get(size_field) == 0)
                price_field = fields.get(name.removesuffix("_size"))
                absent_size = name in ("bid_size", "ask_size") and (row.get(price_field) is None or pd.isna(row.get(price_field)))
                event[name] = number(row[original], original, invalid_fields, allow_missing=absent_side or absent_size)
        if kind == "quote":
            event["source_price"] = number(row.get("price"), "price", invalid_fields, allow_missing=True)
            event["source_size"] = number(row.get("size"), "size", invalid_fields)
            if event.get("action") == "T":
                event["trade_price"] = event["source_price"]
                event["trade_size"] = event["source_size"]
            bid, ask = event.get("bid"), event.get("ask")
            bid_size, ask_size = event.get("bid_size"), event.get("ask_size")
            valid_sides = all(value is not None and value > 0 for value in (bid, ask, bid_size, ask_size)) and bid <= ask
            event["price"] = bid / 2 + ask / 2 if valid_sides else None
        else:
            event["price"] = number(row.get("price"), "price", invalid_fields)
        if invalid_fields:
            event["invalid_fields"] = invalid_fields
        available = pd.Timestamp(row.get("ts_recv", timestamp))
        available = available.tz_localize("UTC") if available.tzinfo is None else available.tz_convert("UTC")
        event["available_timestamp"] = available.isoformat()
        if "ts_recv" in row:
            event["ts_recv"] = available.isoformat()
        events.append(event)
    starts = [event["available_timestamp"] for event in events]
    start, end = min(starts, key=pd.Timestamp), max(starts, key=pd.Timestamp)
    duration = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()
    metadata = {**metadata, "schema": schema, "symbols": sorted({event["symbol"] for event in events}), "records": len(events), "start": start, "end": end, "duration_seconds": duration, "asset_class": "equity", "price_basis": "Quote midpoint; source_price retains the underlying book event price" if kind == "quote" else "Individual executed trade price"}
    return MarketData(events, metadata)


def load_data(payload, root=PROJECT_ROOT):
    request = validate_request(payload)
    root = Path(root).resolve()
    source = request["source"]
    metadata = {"source": source}
    if source == "bundled":
        folder = root / "sample_data"
        sample = request.get("sample", "quotes")
        provenance = json.loads((folder / ("quotes_provenance.json" if sample == "quotes" else "trades_provenance.json")).read_text())
        path = folder / provenance["file"]
        metadata.update(dataset=provenance["dataset"], schema=provenance["schema"], label="AAPL · Level-1 quotes · September 15, 2025" if sample == "quotes" else "AAPL + SPY · Individual trades · September 15, 2025", sample=sample, provenance=provenance)
    elif source == "local":
        path = local_path(request["path"], root)
        manifest_path = path.parent / "manifest.json"
        if manifest_path.is_file() and manifest_path.resolve().is_relative_to((root / "data").resolve()):
            manifest = json.loads(manifest_path.read_text())
            _merge_context(metadata, manifest)
            _merge_context(metadata, manifest.get("request", {}))
            if path.stem.startswith("bars_") and manifest.get("output_schema"):
                metadata["schema"] = manifest["output_schema"]
        metadata["label"] = str(path.relative_to(root / "data"))
    else:
        config = load_config(root)
        client = create_historical_client(config)
        now = datetime.now(timezone.utc)
        folder = root / "data" / "downloads" / f"{now:%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
        folder.mkdir(parents=True)
        path = folder / "data.dbn.zst"
        parameters = {key: value for key, value in request.items() if key != "source"}
        manifest = {"status": "loading", "mode": "historical", "request": parameters, "timezone": TIMEZONE}
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        try:
            store = client.timeseries.get_range(**parameters, path=path)
            try:
                store.to_parquet(folder / "data.parquet")
                frame = _bounded_store_frame(store)
            finally:
                store.reader.close()
            manifest.update(status="complete", records=len(frame))
        except Exception as exc:
            manifest.update(status="failed", error_type=type(exc).__name__)
            if isinstance(exc, ValueError) and str(exc).startswith("The detector accepts"):
                raise
            raise ValueError(f"Databento download failed ({type(exc).__name__}). Check credentials, dataset entitlement, symbols, and date range.") from None
        finally:
            (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        metadata.update(dataset=request["dataset"], schema=request["schema"], label=f"Databento · {', '.join(request['symbols'])}", path=str(path.relative_to(root / "data")))
        return normalize_frame(frame, metadata)
    frame = _read_file(path, metadata)
    return normalize_frame(frame, metadata)
