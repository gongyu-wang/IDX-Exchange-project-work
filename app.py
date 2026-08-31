from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "trained_pipeline.joblib"
DATA_PATH = ROOT / "crmls_sfr_quality_cleaned_202501_202605.csv"

REQUIRED_COLUMNS = [
    "LivingArea",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "LotSizeSquareFeet",
    "City",
    "CountyOrParish",
    "PostalCode",
]

st.set_page_config(
    page_title="California Home Price Analytics",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return pd.DataFrame()

    data = pd.read_csv(DATA_PATH, low_memory=False)

    numeric_columns = [
        "ClosePrice",
        "ListPrice",
        "LivingArea",
        "BedroomsTotal",
        "BathroomsTotalInteger",
        "LotSizeSquareFeet",
        "DaysOnMarket",
        "Latitude",
        "Longitude",
        "PostalCode",
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    if "CloseDate" in data.columns:
        data["CloseDate"] = pd.to_datetime(data["CloseDate"], errors="coerce")

    if "ClosePrice" in data.columns and "LivingArea" in data.columns:
        data["PricePerSqFt"] = data["ClosePrice"] / data["LivingArea"].replace(0, np.nan)

    return data


def money(value):
    if pd.isna(value):
        return "N/A"
    return f"${value:,.0f}"


def page_title(title, description):
    st.title(title)
    st.caption(description)


def sorted_text_values(data, column):
    if column not in data.columns:
        return []
    return sorted(data[column].dropna().astype(str).str.strip().unique().tolist())


def prediction_page(data):
    page_title(
        "🏠 Home Price Prediction",
        "Enter the property characteristics to estimate its likely closing price.",
    )

    model = load_model()
    if model is None:
        st.error(
            "trained_pipeline.joblib was not found. Put the saved deployment pipeline "
            "in the same folder as app.py."
        )
        return

    with st.form("prediction_form"):
        st.subheader("Required property information")
        col1, col2 = st.columns(2)
        with col1:
            living_area = st.number_input(
                "Living area (sq ft)", min_value=100, max_value=30000, value=1800, step=50
            )
            bedrooms = st.number_input(
                "Bedrooms", min_value=0, max_value=20, value=3, step=1
            )
        with col2:
            bathrooms = st.number_input(
                "Bathrooms", min_value=0.0, max_value=20.0, value=2.0, step=0.5
            )
            lot_size = st.number_input(
                "Lot size (sq ft)", min_value=0, max_value=1000000, value=6000, step=100
            )

        st.subheader("Optional location information")
        use_location = st.checkbox(
            "Add location information",
            value=True,
            help="Location can improve the prediction when the city, county, or ZIP appeared in training data.",
        )

        city = None
        county = None
        postal_code = None

        if use_location:
            loc1, loc2, loc3 = st.columns(3)
            with loc1:
                county_options = ["Not specified"] + sorted_text_values(data, "CountyOrParish")
                county = st.selectbox("County", county_options)

            location_data = data
            if county != "Not specified" and "CountyOrParish" in data.columns:
                location_data = data[data["CountyOrParish"].astype(str).eq(county)]

            with loc2:
                city_options = ["Not specified"] + sorted_text_values(location_data, "City")
                city = st.selectbox("City", city_options)

            zip_data = location_data
            if city != "Not specified" and "City" in zip_data.columns:
                zip_data = zip_data[zip_data["City"].astype(str).eq(city)]

            with loc3:
                zip_values = []
                if "PostalCode" in zip_data.columns:
                    zip_values = sorted(
                        zip_data["PostalCode"].dropna().astype(int).astype(str).unique().tolist()
                    )
                postal_code = st.selectbox("ZIP code", ["Not specified"] + zip_values)

        submitted = st.form_submit_button("Predict price", type="primary")

    if submitted:
        input_data = pd.DataFrame(
            {
                "LivingArea": [living_area],
                "BedroomsTotal": [bedrooms],
                "BathroomsTotalInteger": [bathrooms],
                "LotSizeSquareFeet": [lot_size],
                "City": [None if city in (None, "Not specified") else city],
                "CountyOrParish": [None if county in (None, "Not specified") else county],
                "PostalCode": [None if postal_code in (None, "Not specified") else str(postal_code)],
            }
        )

        try:
            prediction = float(model.predict(input_data[REQUIRED_COLUMNS])[0])
        except Exception as error:
            st.error("The prediction pipeline and the App input columns do not match.")
            st.exception(error)
            return

        st.divider()
        result1, result2 = st.columns(2)
        result1.metric("Estimated closing price", money(prediction))
        result2.metric("Estimated price per sq ft", money(prediction / living_area))

        st.info(
            "This estimate is a screening reference, not a formal appraisal. Unusual, luxury, "
            "or highly location-specific properties should be reviewed using comparable sales."
        )


def geographic_page(data):
    page_title(
        "📍 Geographic Market Analysis",
        "Compare transaction activity and price levels across California ZIP codes.",
    )

    needed = {"ClosePrice", "LivingArea", "Latitude", "Longitude", "PostalCode"}
    if data.empty or not needed.issubset(data.columns):
        st.warning("The cleaned data file is missing one or more columns required for this page.")
        return

    county_options = ["All counties"] + sorted_text_values(data, "CountyOrParish")
    filter_col, metric_col = st.columns(2)
    with filter_col:
        selected_county = st.selectbox("County", county_options)
    with metric_col:
        metric_choice = st.selectbox(
            "Map color",
            ["Median sale price", "Median price per sq ft", "Median days on market"],
        )

    filtered = data.copy()
    if selected_county != "All counties":
        filtered = filtered[filtered["CountyOrParish"].astype(str).eq(selected_county)]

    if filtered.empty:
        st.warning("No transactions are available for this selection.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Homes sold", f"{len(filtered):,}")
    k2.metric("Median sale price", money(filtered["ClosePrice"].median()))
    k3.metric("Median price / sq ft", money(filtered["PricePerSqFt"].median()))
    if "DaysOnMarket" in filtered.columns:
        median_dom = filtered["DaysOnMarket"].median()
        k4.metric("Median days on market", "N/A" if pd.isna(median_dom) else f"{median_dom:.0f}")
    else:
        k4.metric("Median days on market", "N/A")

    aggregations = {
        "City": "first",
        "ClosePrice": ["size", "median"],
        "PricePerSqFt": "median",
        "Latitude": "median",
        "Longitude": "median",
    }
    if "CountyOrParish" in filtered.columns:
        aggregations["CountyOrParish"] = "first"
    if "DaysOnMarket" in filtered.columns:
        aggregations["DaysOnMarket"] = "median"

    zip_summary = (
        filtered.dropna(subset=["PostalCode", "Latitude", "Longitude"])
        .groupby("PostalCode")
        .agg(aggregations)
    )
    zip_summary.columns = [
        "_".join(part for part in column if part).strip("_")
        if isinstance(column, tuple)
        else column
        for column in zip_summary.columns
    ]
    zip_summary = zip_summary.reset_index().rename(
        columns={
            "City_first": "City",
            "CountyOrParish_first": "County",
            "ClosePrice_size": "Homes",
            "ClosePrice_median": "MedianPrice",
            "PricePerSqFt_median": "MedianPPSF",
            "DaysOnMarket_median": "MedianDOM",
            "Latitude_median": "Latitude",
            "Longitude_median": "Longitude",
        }
    )
    zip_summary = zip_summary[
        (zip_summary["Homes"] >= 5)
        & zip_summary["Latitude"].between(32.0, 42.5)
        & zip_summary["Longitude"].between(-125.0, -113.0)
    ].copy()

    metric_columns = {
        "Median sale price": "MedianPrice",
        "Median price per sq ft": "MedianPPSF",
        "Median days on market": "MedianDOM",
    }
    color_column = metric_columns[metric_choice]

    if color_column not in zip_summary.columns or zip_summary.empty:
        st.info("There are not enough ZIP-level records to draw this map.")
        return

    zip_summary = zip_summary.dropna(subset=[color_column])
    percentile = zip_summary[color_column].rank(pct=True)
    zip_summary["Color"] = [
        [37, 99, 235, 180] if p <= 0.33 else [15, 118, 110, 180] if p <= 0.67 else [220, 80, 45, 180]
        for p in percentile
    ]
    zip_summary["Radius"] = np.sqrt(zip_summary["Homes"]) * 500
    zip_summary["ZIP"] = zip_summary["PostalCode"].astype(int).astype(str).str.zfill(5)
    zip_summary["MedianPriceLabel"] = zip_summary["MedianPrice"].map(money)
    zip_summary["MedianPPSFLabel"] = zip_summary["MedianPPSF"].map(money)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=zip_summary,
        get_position="[Longitude, Latitude]",
        get_fill_color="Color",
        get_radius="Radius",
        radius_min_pixels=4,
        radius_max_pixels=30,
        pickable=True,
    )
    view = pdk.ViewState(
        latitude=float(zip_summary["Latitude"].median()),
        longitude=float(zip_summary["Longitude"].median()),
        zoom=5.2 if selected_county == "All counties" else 8,
    )
    tooltip = {
        "html": "<b>ZIP {ZIP}</b> · {City}<br/>Homes: {Homes}<br/>Median price: {MedianPriceLabel}<br/>Median $/sq ft: {MedianPPSFLabel}",
        "style": {"backgroundColor": "#172033", "color": "white"},
    }
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip))
    st.caption(
        f"Each circle represents a ZIP code with at least five sales. Circle size represents transaction volume; color represents {metric_choice.lower()}."
    )

    st.subheader("Largest city markets")
    if "City" in filtered.columns:
        city_summary = (
            filtered.dropna(subset=["City"])
            .groupby("City")
            .agg(
                Homes=("ClosePrice", "size"),
                MedianPrice=("ClosePrice", "median"),
                MedianPPSF=("PricePerSqFt", "median"),
            )
            .sort_values("Homes", ascending=False)
            .head(15)
            .reset_index()
        )
        city_summary["MedianPrice"] = city_summary["MedianPrice"].map(money)
        city_summary["MedianPPSF"] = city_summary["MedianPPSF"].map(money)
        st.dataframe(city_summary, hide_index=True, use_container_width=True)


def trends_page(data):
    page_title(
        "📈 Market Trends",
        "Track monthly transaction volume, median sale price, and median price per square foot.",
    )

    needed = {"CloseDate", "ClosePrice", "PricePerSqFt"}
    if data.empty or not needed.issubset(data.columns):
        st.warning("The cleaned data file is missing one or more columns required for this page.")
        return

    county_options = ["All counties"] + sorted_text_values(data, "CountyOrParish")
    selected_county = st.selectbox("County", county_options, key="trend_county")

    filtered = data.dropna(subset=["CloseDate", "ClosePrice"]).copy()
    if selected_county != "All counties":
        filtered = filtered[filtered["CountyOrParish"].astype(str).eq(selected_county)]

    filtered["Month"] = filtered["CloseDate"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        filtered.groupby("Month")
        .agg(
            HomesSold=("ClosePrice", "size"),
            MedianSalePrice=("ClosePrice", "median"),
            MedianPricePerSqFt=("PricePerSqFt", "median"),
        )
        .reset_index()
        .sort_values("Month")
    )

    if monthly.empty:
        st.warning("No monthly transactions are available for this selection.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Transactions in view", f"{len(filtered):,}")
    m2.metric("Overall median price", money(filtered["ClosePrice"].median()))
    m3.metric("Overall median $/sq ft", money(filtered["PricePerSqFt"].median()))

    st.subheader("Monthly homes sold")
    st.line_chart(monthly.set_index("Month")["HomesSold"], color="#7c3aed")
    st.subheader("Monthly median sale price")
    st.line_chart(monthly.set_index("Month")["MedianSalePrice"], color="#2563eb")
    st.subheader("Monthly median price per square foot")
    st.line_chart(monthly.set_index("Month")["MedianPricePerSqFt"], color="#0f766e")
    st.caption("These charts describe historical closed transactions and do not establish causal market drivers.")


data = load_data()

st.sidebar.title("California Home Analytics")
page = st.sidebar.radio(
    "Navigation",
    ["Home Price Prediction", "Geographic Market Analysis", "Market Trends"],
)
st.sidebar.divider()
st.sidebar.caption("CRMLS single-family residential analysis")

if page == "Home Price Prediction":
    prediction_page(data)
elif page == "Geographic Market Analysis":
    geographic_page(data)
else:
    trends_page(data)
