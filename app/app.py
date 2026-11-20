"""AgriSense - Streamlit app.   Run:  streamlit run app/app.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src import risk
from src.predict import get_predictor, spi_label

st.set_page_config(page_title="AgriSense", page_icon="🌾", layout="wide")


@st.cache_resource(show_spinner="Loading model and data...")
def load():
    return get_predictor()


@st.cache_data
def load_risk():
    ex = ROOT / "exports"
    return (pd.read_csv(ex / "district_crop_risk.csv"), pd.read_csv(ex / "district_risk_tiers.csv"),
            risk.get_ratios())


@st.cache_data
def load_geojson():
    return json.loads((C.PROCESSED / "districts_simplified.geojson").read_text())


@st.cache_data
def load_results():
    p = C.REPORTS / "model_results.json"
    return json.loads(p.read_text()) if p.exists() else {}


P = load()
dcr, dtier, ratios = load_risk()

st.sidebar.title("🌾 AgriSense")
st.sidebar.caption("Crop yield prediction & agricultural risk analysis for Indian districts (ICRISAT data, 1966-2017)")
page = st.sidebar.radio("Page", ["Predict & what-if", "Risk profile", "Risk map", "Model & method"])


def pick_district(key):
    states = sorted(P.dist[P.dist.dist_code.isin(P.available.dist_code)].state_name.unique())
    state = st.selectbox("State", states, index=states.index("Maharashtra") if "Maharashtra" in states else 0,
                         key=f"s{key}")
    d = P.dist[(P.dist.state_name == state) & P.dist.dist_code.isin(P.available.dist_code)].sort_values("dist_name")
    name = st.selectbox("District", d.dist_name.tolist(), key=f"d{key}")
    return int(d[d.dist_name == name].dist_code.iloc[0]), name


# =====================================================================================
if page == "Predict & what-if":
    st.title("Yield prediction & rainfall what-if")
    st.caption("Replays a historical season with the trained model. Change the monsoon rainfall to ask: "
               "*what would the model have expected if the monsoon had been different?*")
    c1, c2, c3 = st.columns(3)
    with c1:
        dist_code, dist_name = pick_district("p")
    crops = P.crops_for(dist_code)
    with c2:
        crop = st.selectbox("Crop", crops, index=crops.index("Rice") if "Rice" in crops else 0)
        years = [y for y in P.years_for(dist_code, crop) if y >= 1975]
        year = st.select_slider("Year", options=years, value=years[-1] if years else None)
    with c3:
        factor = st.slider("Monsoon (Jun-Sep) rainfall vs what actually fell", 40, 160, 100, 5,
                           format="%d%%") / 100

    try:
        res = P.predict(dist_code, crop, int(year), factor)
    except ValueError as e:
        st.warning(f"Cannot predict this combination: {e}")
        st.stop()

    if not res["weather_available"]:
        st.warning("No rainfall record for this district-year: the model is running without weather features.")
    a, b, c, d = st.columns(4)
    a.metric("Model prediction", f"{res['predicted_kg_ha']:,.0f} kg/ha",
             f"{(res['predicted_kg_ha']/res['baseline_kg_ha']-1)*100:+.1f}% vs 5-yr average")
    b.metric("80% interval", f"{res['lo80_kg_ha']:,.0f} - {res['hi80_kg_ha']:,.0f}")
    c.metric("5-year average (baseline)", f"{res['baseline_kg_ha']:,.0f} kg/ha")
    d.metric("Actual yield", "-" if res["actual_kg_ha"] is None else f"{res['actual_kg_ha']:,.0f} kg/ha")

    if res["spi_monsoon"] is not None:
        st.info(f"Monsoon in this scenario: **{res['monsoon_ratio']*100:.0f}%** of the district's 1966-89 average "
                f"(SPI {res['spi_monsoon']:+.2f}, *{spi_label(res['spi_monsoon'])}*).")
    if res["in_sample"]:
        st.warning(f"⚠️ {res['year']} was part of the data the final model was trained on (up to "
                   f"{P.trained_through}), so a close match here proves little. Years 2010-2015 are out-of-sample.")
    else:
        st.success(f"✅ {res['year']} is out-of-sample: the model never saw this year.")

    h = P.crop[(P.crop.dist_code == dist_code) & (P.crop.crop == crop) & (P.crop.year <= 2015)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.year, y=h.yield_kg_ha, mode="lines+markers", name="Actual yield",
                             line=dict(color="#2a9d8f")))
    fig.add_trace(go.Scatter(x=[res["year"], res["year"]], y=[res["lo80_kg_ha"], res["hi80_kg_ha"]],
                             mode="lines", line=dict(color="#d62828", width=6), name="80% interval"))
    fig.add_trace(go.Scatter(x=[res["year"]], y=[res["predicted_kg_ha"]], mode="markers",
                             marker=dict(color="#d62828", size=13, symbol="diamond"), name="Prediction"))
    fig.update_layout(title=f"{crop} in {dist_name}: yield history", yaxis_title="kg / ha", height=380,
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The what-if changes only Jun-Sep rainfall of the chosen year (all monsoon-derived features are "
               "recomputed). It shows the model's learned association, not a causal or irrigation-aware simulation.")

# =====================================================================================
elif page == "Risk profile":
    st.title("District-crop risk profile")
    c1, c2 = st.columns(2)
    with c1:
        dist_code, dist_name = pick_district("r")
    avail = sorted(dcr[dcr.dist_code == dist_code].crop.unique())
    if not avail:
        st.warning("Not enough history for any crop in this district (needs >= 20 years).")
        st.stop()
    with c2:
        crop = st.selectbox("Crop", avail, key="rc")
    m = dcr[(dcr.dist_code == dist_code) & (dcr.crop == crop)].iloc[0]
    tier = dtier[dtier.dist_code == dist_code]
    k = st.columns(5)
    k[0].metric("Risk score (0-100)", f"{m.risk_score:.0f}")
    k[1].metric("Yield volatility", f"{m.cv_detrended*100:.0f}%")
    k[2].metric("Years >15% below trend", f"{m.downside_freq*100:.0f}%")
    k[3].metric("Avg loss in worst 10% yrs", f"{m.cvar10_loss*100:.0f}%")
    k[4].metric("District tier", tier.risk_tier.iloc[0] if len(tier) else "n/a")
    if pd.notna(m.drought_loss):
        st.write(f"In **{int(m.drought_years_n)} drought monsoons** (SPI < -1) this crop lost on average "
                 f"**{m.drought_loss*100:.0f}%** vs its local trend.")

    st.subheader("Monte Carlo: next-season outcomes")
    s1, s2, s3 = st.columns(3)
    scenario = s1.radio("Scenario", ["normal", "drought"], horizontal=True,
                        help="'drought' resamples only years with SPI < -1")
    area = s2.number_input("Area (hectares)", 0.5, 1000.0, 1.0)
    price = s3.number_input("Price (Rs per kg, optional)", 0.0, 500.0, 0.0, help="You supply this: no price data is bundled.")
    sim = risk.simulate(ratios, dist_code, crop, scenario=scenario, area_ha=area,
                        price_per_kg=price if price > 0 else None)
    fig = px.histogram(x=sim["samples"], nbins=60, labels={"x": "Simulated yield (kg/ha)"},
                       color_discrete_sequence=["#2a9d8f" if scenario == "normal" else "#d62828"])
    fig.add_vline(x=sim["baseline_kg_ha"], line_dash="dash", annotation_text="5-yr average")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    q = st.columns(5)
    for col, (lab, key) in zip(q, [("5th pct", "p5"), ("Median", "p50"), ("95th pct", "p95"),
                                   ("P(yield < 80% of avg)", "prob_below_80pct_of_baseline"),
                                   ("Avg of worst 5%", "cvar5_kg_ha")]):
        v = sim[key]
        col.metric(lab, f"{v*100:.0f}%" if key.startswith("prob") else f"{v:,.0f} kg/ha")
    if "revenue" in sim:
        r = sim["revenue"]
        st.write(f"Revenue on {area:g} ha at Rs {price:g}/kg: median **Rs {r['p50']:,.0f}**, "
                 f"5th percentile **Rs {r['p5']:,.0f}**, 95th percentile **Rs {r['p95']:,.0f}**.")
    st.caption(f"Resampled from: {sim['pool']} ({sim['pool_size']} historical yield ratios). Past variability is "
               "assumed to repeat; irrigation changes, new varieties and climate shifts are not modelled.")

# =====================================================================================
elif page == "Risk map":
    st.title("District risk map")
    metric = st.radio("Colour by", ["risk_tier", "risk_score", "drought_loss"], horizontal=True,
                      format_func=lambda x: {"risk_tier": "Risk tier", "risk_score": "Risk score",
                                             "drought_loss": "Avg drought-year loss"}[x])
    gj = load_geojson()
    df = dtier.copy()
    if metric == "risk_tier":
        fig = px.choropleth(df, geojson=gj, locations="dist_code", featureidkey="properties.dist_code",
                            color="risk_tier", color_discrete_map={"Low": "#2a9d8f", "Medium": "#e9c46a", "High": "#d62828"},
                            category_orders={"risk_tier": ["Low", "Medium", "High"]}, hover_name="dist_name",
                            hover_data={"state_name": True, "risk_score": ":.0f", "dist_code": False})
    else:
        fig = px.choropleth(df, geojson=gj, locations="dist_code", featureidkey="properties.dist_code",
                            color=metric, color_continuous_scale="YlOrRd", hover_name="dist_name",
                            hover_data={"state_name": True, "dist_code": False})
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=640, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Districts use ICRISAT's fixed 1966 boundaries; districts split later are approximated by their "
               "modern parts. Hill and tiny districts can look risky partly because their yield records are noisy.")
    cA, cB = st.columns(2)
    cA.subheader("Highest risk"); cA.dataframe(df.sort_values("risk_score", ascending=False).head(10)
                                              [["dist_name", "state_name", "risk_score"]], hide_index=True)
    cB.subheader("Lowest risk"); cB.dataframe(df.sort_values("risk_score").head(10)
                                             [["dist_name", "state_name", "risk_score"]], hide_index=True)

# =====================================================================================
else:
    st.title("Model & method")
    R = load_results()
    if R:
        sel = R["selected"]
        te = R["final_refit_test"]["weather_complete"]
        cov = R["coverage"]
        k = st.columns(4)
        k[0].metric("Selected model", sel)
        k[1].metric("Test MAE skill vs baseline", f"{te['skill_MAE_pct']:+.1f}%")
        k[2].metric("Test median error", f"{te['MedAPE_pct']:.1f}%")
        k[3].metric("80% interval coverage (test)", f"{cov['test_weather_complete']*100:.0f}%")
        abl = json.loads((C.REPORTS / "ablation.json").read_text()) if (C.REPORTS / "ablation.json").exists() else {}
        st.markdown(f"""
**How to read this.** The baseline predicts a district's yield as the mean of its previous five years. The model
predicts the *deviation* from that baseline using weather (rainfall anomalies, SPI), yield history and location.
Skill is the reduction in mean absolute error versus that baseline on **2010-2015 data the model never saw**
(rows with rainfall records). On those years the model reduces error by **{te['skill_MAE_pct']:.1f}%** and explains
about **{te['R2_deviation']*100:.0f}%** of the year-to-year deviation from the local norm. That is a modest, honest gain:
yields depend heavily on irrigation, inputs and management that this dataset does not contain. The 80% intervals
actually covered **{cov['test_weather_complete']*100:.0f}%** of test outcomes (slightly optimistic, not perfect).
""")
        if abl:
            st.markdown("**Does weather help?** (same model, different inputs, test 2010-15)")
            st.dataframe(pd.DataFrame(abl).T[["n_features", "valid_skill_pct", "test_skill_pct", "test_R2_deviation"]].round(3))
            st.caption("Weather roughly doubles test skill but added little on the 2005-09 validation window, "
                       "so its exact contribution is uncertain.")
        cmp_ = pd.DataFrame({m: {"valid skill %": v["valid"]["weather_complete"]["skill_MAE_pct"],
                                 "test skill %": v["test"]["weather_complete"]["skill_MAE_pct"],
                                 "test MAE kg/ha": v["test"]["weather_complete"]["MAE"]}
                             for m, v in R["comparison"].items()}).T.round(2)
        st.dataframe(cmp_)
    figs = C.REPORTS / "figures"
    cols = st.columns(2)
    for col, (f, cap) in zip(cols * 2, [("shap_importance.png", "Global SHAP importance"),
                                        ("model_skill_by_crop.png", "Skill by crop"),
                                        ("risk_map.png", "Risk tiers"), ("model_comparison.png", "Model comparison")]):
        if (figs / f).exists():
            col.image(str(figs / f), caption=cap)
    st.markdown("""
**Data:** ICRISAT District-Level Database (crop area, production, yield, monthly rainfall; 311 districts, 1966-2017).
**Limits:** no irrigation, soil or price data; rainfall ends 2015 and is missing for 22 districts and for ~87 districts
in 2010-11; district boundaries are 1966-based; what-if is associational, not causal.
""")
