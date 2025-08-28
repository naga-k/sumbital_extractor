"""
Fire Alarm Submittal Extractor (Spec + Submittals → SQLite)

This variant uses only the OpenAI Responses API (gpt-5-mini) to analyze uploaded PDFs.
All local regex/text extraction paths have been removed; uploaded PDFs are sent
directly to the LLM for extraction.
"""
import os
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple
import time
import ssl
import httpx

import streamlit as st
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

# Optional LLM (Responses API client)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    _OPENAI_CLIENT: Optional[OpenAI] = OpenAI()  # uses OPENAI_API_KEY from env
except ImportError:
    OPENAI_AVAILABLE = False
    _OPENAI_CLIENT = None

# ----------------------------
# Data Models
# ----------------------------

@dataclass
class SpecRequirement:
    spec_key: str
    clause_text: str
    requirement_type: str

@dataclass
class DeviceItem:
    model: str
    description: str
    qty: int

@dataclass
class BatteryCalcRow:
    panel_name: str
    item_desc: str
    model: str
    qty: int
    supervis_current_a: float
    alarm_current_a: float

@dataclass
class BatteryPanelSummary:
    panel_name: str
    total_amp_hours_required: float
    provided_batteries: str

@dataclass
class VoltageDropEntry:
    circuit_name: str
    conductor_awg: Optional[str]
    length_ft: Optional[float]
    voltage_drop_pct: Optional[float]
    eol_voltage_v: Optional[float]

# ----------------------------
# Optional LLM extraction (Responses API)
# ----------------------------

def _call_openai_json(system_prompt: str, user_payload: str, file_bytes: Optional[bytes] = None, response_hint: str = "") -> Optional[str]:
    """Use the OpenAI Responses API client to upload an optional file and request strict JSON output."""
    if not OPENAI_AVAILABLE or _OPENAI_CLIENT is None:
        st.error("OpenAI Responses client is not available. Install openai>=1.0.0 and set OPENAI_API_KEY.")
        return None

    # Allow Streamlit secrets to set env var for the official client
    if not os.getenv("OPENAI_API_KEY") and hasattr(st, "secrets"):
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            os.environ["OPENAI_API_KEY"] = key

    client = _OPENAI_CLIENT

    def _retry(func, attempts=4, base_delay=1.0):
        last_exc = None
        for i in range(attempts):
            try:
                return func()
            except Exception as e:
                last_exc = e
                delay = base_delay * (2 ** i)
                print(f"Transient error ({type(e).__name__}): {e}. retry {i+1}/{attempts} after {delay:.1f}s", flush=True)
                time.sleep(delay)
        # re-raise final exception
        raise last_exc

    start_total = time.time()
    file_id = None
    upload_dur = resp_dur = None
    if file_bytes:
        import io
        start_upload = time.time()
        fobj = io.BytesIO(file_bytes)
        fobj.name = "upload.pdf"
        try:
            created = _retry(lambda: client.files.create(file=fobj, purpose="assistants"))
            file_id = created.id
        except Exception as e:
            print(f"Failed to upload file after retries: {e}", flush=True)
            st.error(f"Failed to upload file to OpenAI: {e}")
            return None
        upload_dur = time.time() - start_upload

    try:
        start_resp = time.time()
        # Build multimodal input for Responses API (input parts)
        content_parts = [{"type": "input_text", "text": (user_payload + (" " + response_hint if response_hint else ""))}]
        if file_id:
            content_parts.append({"type": "input_file", "file_id": file_id})

        try:
            resp = _retry(lambda: client.responses.create(
                model="gpt-5-mini",
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": content_parts},
                ],
                temperature=1,
            ))
        except Exception as e:
            print(f"Failed to create response after retries: {e}", flush=True)
            st.error(f"LLM request failed after retries: {e}")
            return None

        resp_dur = time.time() - start_resp

        # prefer output_text convenience property; fallback to parsing response output
        out_text = getattr(resp, "output_text", None)
        total = time.time() - start_total
        print(f"LLM timing: upload={upload_dur or 0:.2f}s resp={resp_dur:.2f}s total={total:.2f}s", flush=True)
        return out_text
    except Exception as e:
        st.error(f"LLM call failed: {e}")
        return None
    finally:
        if file_id:
            try:
                client.files.delete(file_id)
            except Exception:
                pass

# LLM wrapper functions (accept PDF bytes and return structured dataclasses)

def llm_extract_spec_items(file_bytes: bytes) -> List[SpecRequirement]:
    sys_p = (
        "You extract compliance checklists from construction specifications (NFPA 72 fire alarm) in the uploaded PDF. "
        "Output JSON array where each element has: spec_key (slug), clause_text (verbatim excerpt), requirement_type "
        "in {battery, voltage_drop, wiring, records, quality, hardware, capacity, programming, documentation, other}."
    )
    usr = "Analyze the uploaded PDF and extract the spec requirements as specified. Return strict JSON."
    js = _call_openai_json(sys_p, usr, file_bytes=file_bytes, response_hint="Respond with strict JSON array only.")
    out: List[SpecRequirement] = []
    if js:
        try:
            data = json.loads(js)
            for item in data:
                spec_key = str(item.get("spec_key", "")).strip() or "item"
                clause_text = str(item.get("clause_text", ""))[:2000]
                requirement_type = str(item.get("requirement_type", "other"))
                out.append(SpecRequirement(spec_key=spec_key, clause_text=clause_text, requirement_type=requirement_type))
        except Exception:
            st.error("Failed to parse spec JSON from LLM.")
    return out

def llm_extract_battery(file_bytes: bytes) -> Tuple[List[BatteryCalcRow], List[BatteryPanelSummary]]:
    sys_p = (
        "You parse fire alarm battery calculation PDFs into structured JSON. "
        "Return an object with 'rows' and 'panels'. 'rows' is an array of {panel_name, item_desc, model, qty, supervis_current_a, alarm_current_a}. "
        "'panels' is an array of {panel_name, total_amp_hours_required, provided_batteries}."
    )
    usr = "Analyze the uploaded PDF and extract battery calculation data as specified. Return strict JSON."
    js = _call_openai_json(sys_p, usr, file_bytes=file_bytes, response_hint="Return strict JSON object with keys 'rows' and 'panels'.")
    rows: List[BatteryCalcRow] = []
    sums: List[BatteryPanelSummary] = []
    if js:
        try:
            data = json.loads(js)
            for r in data.get("rows", []):
                try:
                    rows.append(BatteryCalcRow(
                        panel_name=str(r.get("panel_name", "")).strip(),
                        item_desc=str(r.get("item_desc", "")).strip(),
                        model=str(r.get("model", "")).strip(),
                        qty=int(float(r.get("qty", 0))),
                        supervis_current_a=float(r.get("supervis_current_a", 0.0)),
                        alarm_current_a=float(r.get("alarm_current_a", 0.0)),
                    ))
                except Exception:
                    continue
            for p in data.get("panels", []):
                try:
                    sums.append(BatteryPanelSummary(
                        panel_name=str(p.get("panel_name", "")).strip(),
                        total_amp_hours_required=float(p.get("total_amp_hours_required", 0.0)),
                        provided_batteries=str(p.get("provided_batteries", "")).strip(),
                    ))
                except Exception:
                    continue
        except Exception:
            st.error("Failed to parse battery JSON from LLM.")
    return rows, sums

def llm_extract_vdrop(file_bytes: bytes) -> List[VoltageDropEntry]:
    sys_p = (
        "You parse notification/appliance circuit voltage-drop schedules into JSON. "
        "Return an array of entries with {circuit_name, conductor_awg, length_ft, voltage_drop_pct, eol_voltage_v}."
    )
    usr = "Analyze the uploaded PDF and extract voltage drop data as specified. Return strict JSON array."
    js = _call_openai_json(sys_p, usr, file_bytes=file_bytes, response_hint="Respond with strict JSON array only.")
    out: List[VoltageDropEntry] = []
    if js:
        try:
            data = json.loads(js)
            for e in data:
                try:
                    out.append(VoltageDropEntry(
                        circuit_name=str(e.get("circuit_name", "")).strip(),
                        conductor_awg=str(e.get("conductor_awg", "")).strip() or None,
                        length_ft=float(e.get("length_ft", 0.0)) if e.get("length_ft") is not None else None,
                        voltage_drop_pct=float(e.get("voltage_drop_pct", 0.0)) if e.get("voltage_drop_pct") is not None else None,
                        eol_voltage_v=float(e.get("eol_voltage_v", 0.0)) if e.get("eol_voltage_v") is not None else None,
                    ))
                except Exception:
                    continue
        except Exception:
            st.error("Failed to parse voltage-drop JSON from LLM.")
    return out

def llm_extract_shop_devices(file_bytes: bytes) -> List[DeviceItem]:
    sys_p = (
        "You read fire alarm shop drawings PDFs and enumerate device schedules. "
        "Return JSON array of devices with fields {model, description, qty}. Group identical models."
    )
    usr = "Analyze the uploaded PDF and extract device data as specified. Return strict JSON array."
    js = _call_openai_json(sys_p, usr, file_bytes=file_bytes, response_hint="Respond with strict JSON array only.")
    out: List[DeviceItem] = []
    if js:
        try:
            data = json.loads(js)
            for d in data:
                try:
                    out.append(DeviceItem(
                        model=str(d.get("model", "")).strip(),
                        description=str(d.get("description", "")).strip(),
                        qty=int(float(d.get("qty", 0)))
                    ))
                except Exception:
                    continue
        except Exception:
            st.error("Failed to parse shop devices JSON from LLM.")
    return out

# ----------------------------
# Streamlit UI (LLM-only)
# ----------------------------

st.set_page_config(page_title="FA Submittal Extractor (LLM-only)", layout="wide")
st.title("Fire Alarm Submittal Extractor — LLM-only")

# Initialize session state
for k in ("last_spec", "last_devices", "last_brows", "last_bsum", "last_vdrop"):
    if k not in st.session_state:
        st.session_state[k] = []

with st.sidebar:
    st.header("Upload PDFs (will be sent to OpenAI Responses API)")
    spec_pdf = st.file_uploader("Spec Section (e.g., 283111)", type=["pdf"])
    shop_pdf = st.file_uploader("Shop Drawings (optional)", type=["pdf"])
    battery_pdf = st.file_uploader("Battery Calculations", type=["pdf"])
    vdrop_pdf = st.file_uploader("Voltage Drop Calculations (optional)", type=["pdf"])

if st.button("Extract (LLM only)"):
    if not OPENAI_AVAILABLE:
        st.error("OpenAI Responses client not available. Install openai and set OPENAI_API_KEY.")
    else:
        # Spec
        if spec_pdf is not None:
            spec_bytes = spec_pdf.getvalue()
            spec_llm = llm_extract_spec_items(spec_bytes)
            st.session_state['last_spec'] = [asdict(r) for r in spec_llm]
        else:
            st.session_state['last_spec'] = []

        # Shop drawings
        if shop_pdf is not None:
            shop_bytes = shop_pdf.getvalue()
            devices_llm = llm_extract_shop_devices(shop_bytes)
            st.session_state['last_devices'] = [asdict(d) for d in devices_llm]
        else:
            st.session_state['last_devices'] = []

        # Battery
        if battery_pdf is not None:
            batt_bytes = battery_pdf.getvalue()
            rows_llm, sums_llm = llm_extract_battery(batt_bytes)
            st.session_state['last_brows'] = [asdict(r) for r in rows_llm]
            st.session_state['last_bsum'] = [asdict(s) for s in sums_llm]
        else:
            st.session_state['last_brows'] = []
            st.session_state['last_bsum'] = []

        # Voltage drop
        if vdrop_pdf is not None:
            vdrop_bytes = vdrop_pdf.getvalue()
            vds_llm = llm_extract_vdrop(vdrop_bytes)
            st.session_state['last_vdrop'] = [asdict(e) for e in vds_llm]
        else:
            st.session_state['last_vdrop'] = []

        st.success("LLM extraction complete.")

# Downloads / Display (same as before)
if 'no_db' not in globals():
    no_db = True

if no_db:
    st.subheader("Downloads (JSON)")
    colA, colB, colC = st.columns(3)
    with colA:
        st.download_button("Download spec_requirements.json", data=json.dumps(st.session_state['last_spec'], indent=2), file_name="spec_requirements.json")
        st.download_button("Download devices.json", data=json.dumps(st.session_state['last_devices'], indent=2), file_name="devices.json")
    with colB:
        st.download_button("Download battery_rows.json", data=json.dumps(st.session_state['last_brows'], indent=2), file_name="battery_rows.json")
        st.download_button("Download battery_summary.json", data=json.dumps(st.session_state['last_bsum'], indent=2), file_name="battery_summary.json")
    with colC:
        st.download_button("Download voltage_drop.json", data=json.dumps(st.session_state['last_vdrop'], indent=2), file_name="voltage_drop.json")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Spec requirements")
    df_spec = pd.DataFrame(st.session_state['last_spec'])
    st.dataframe(df_spec, use_container_width=True)

    st.subheader("Devices (from shop drawings)")
    df_dev = pd.DataFrame(st.session_state['last_devices'])
    st.dataframe(df_dev, use_container_width=True)

with col2:
    st.subheader("Battery rows")
    df_brow = pd.DataFrame(st.session_state['last_brows'])
    st.dataframe(df_brow, use_container_width=True)

    st.subheader("Battery panel summaries")
    df_bsum = pd.DataFrame(st.session_state['last_bsum'])
    st.dataframe(df_bsum, use_container_width=True)

    st.subheader("Voltage drop entries")
    df_vd = pd.DataFrame(st.session_state['last_vdrop'])
    st.dataframe(df_vd, use_container_width=True)
