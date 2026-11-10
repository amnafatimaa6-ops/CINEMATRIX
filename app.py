
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import zipfile
import tempfile
import shutil
from pathlib import Path

from statsmodels.tsa.statespace.sarimax import SARIMAX

# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="CINEMETRIX",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 CINEMETRIX")
st.caption(
    "Film Analytics | Audience Intelligence | "
    "Financial Modelling | Predictive AI"
)

GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "History",
    "Horror", "Music", "Mystery", "Romance", "Science Fiction",
    "TV Movie", "Thriller", "War", "Western"
]

MODEL_FILES = {
    "Ridge Revenue": "ridge_revenue.pkl",
    "Random Forest Revenue": "random_forest_revenue.pkl",
    "LightGBM Basic": "lightgbm_revenue.pkl",
    "LightGBM Genre": "lightgbm_genre_revenue.pkl",
    "LightGBM Final": "cinematrix_final_pre_release.pkl",
    "LightGBM Post-release": "lightgbm_post_release.pkl",
    "LightGBM Profit": "lightgbm_profit.pkl",
    "LightGBM Hit Classifier": "lightgbm_hit_classifier.pkl"
}

# =====================================================
# 2. ZIP FILE LOADER
# =====================================================

st.sidebar.header("CINEMETRIX Data")

uploaded_zip = st.sidebar.file_uploader(
    "Upload project backup ZIP",
    type=["zip"]
)

def extract_project(zip_file):

    destination = Path(
        tempfile.mkdtemp(prefix="cinemetrix_")
    )

    required_csvs = {
        "cinemetrix_inflation_adjusted.csv",
        "cinemetrix_model_data.csv",
        "cinemetrix_financial_movies.csv",
        "genre_audience_summary.csv",
        "ratings_by_year.csv",
        "all_model_performance.csv",
        "revenue_model_comparison.csv",
        "sarima_monthly_forecasts.csv",
        "monthly_ratings.csv",
        "financial_scenario_results.csv",
        "top_rated_movies.csv",
        "top_quality_movies.csv",
        "user_activity_summary.csv",
        "final_revenue_predictions.csv",
        "genre_revenue_predictions.csv",
        "shap_summary.csv",
        "final_importance.csv",
        "genre_importance.csv",
        "post_release_feature_importance.csv"
    }

    with zipfile.ZipFile(zip_file, "r") as archive:

        for info in archive.infolist():

            if info.is_dir():
                continue

            filename = Path(info.filename).name

            is_model = (
                filename.endswith(".pkl")
                and "/models/" in "/" + info.filename
            )

            is_csv = filename in required_csvs

            if not (is_model or is_csv):
                continue

            if info.file_size > 250 * 1024 * 1024:
                continue

            target = destination / filename

            with archive.open(info) as source:
                with open(target, "wb") as output:
                    shutil.copyfileobj(source, output)

    return destination


def find_file(folder, filename):

    matches = list(Path(folder).rglob(filename))

    return matches[0] if matches else None


def load_csv(folder, filename):

    path = find_file(folder, filename)

    if path is None:
        return None

    try:
        return pd.read_csv(path)
    except Exception as error:
        st.warning(f"Could not read {filename}: {error}")
        return None


def load_models(folder):

    models = {}

    for name, filename in MODEL_FILES.items():

        path = find_file(folder, filename)

        if path is None:
            continue

        try:
            with open(path, "rb") as file:
                models[name] = pickle.load(file)

        except Exception as error:
            st.warning(f"Could not load {name}: {error}")

    return models


if uploaded_zip is not None:

    if st.session_state.get("zip_name") != uploaded_zip.name:

        with st.spinner("Loading CINEMETRIX project..."):

            zip_path = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".zip"
            )

            zip_path.write(uploaded_zip.getbuffer())
            zip_path.close()

            folder = extract_project(zip_path.name)

            st.session_state["project_folder"] = str(folder)
            st.session_state["zip_name"] = uploaded_zip.name

if "project_folder" not in st.session_state:

    # Optional ZIP stored beside app.py
    local_zip = Path(__file__).parent / "CINEMETRIX_BACKUP.zip"

    if local_zip.exists():

        with st.spinner("Loading project backup..."):

            folder = extract_project(local_zip)

            st.session_state["project_folder"] = str(folder)

if "project_folder" not in st.session_state:

    st.info(
        "Upload your CINEMETRIX backup ZIP using the sidebar."
    )

    st.stop()


PROJECT = Path(st.session_state["project_folder"])

st.sidebar.success("Project loaded successfully")

# =====================================================
# 3. LOAD DATA
# =====================================================

financial = load_csv(
    PROJECT,
    "cinemetrix_inflation_adjusted.csv"
)

if financial is None:
    financial = load_csv(
        PROJECT,
        "cinemetrix_financial_movies.csv"
    )

genre_data = load_csv(
    PROJECT,
    "genre_audience_summary.csv"
)

yearly_ratings = load_csv(
    PROJECT,
    "ratings_by_year.csv"
)

model_results = load_csv(
    PROJECT,
    "all_model_performance.csv"
)

scenario_data = load_csv(
    PROJECT,
    "financial_scenario_results.csv"
)

sarima_saved = load_csv(
    PROJECT,
    "sarima_monthly_forecasts.csv"
)

models = load_models(PROJECT)

# =====================================================
# 4. DASHBOARD TABS
# =====================================================

tabs = st.tabs([
    "Overview",
    "Audience Analytics",
    "Financial Analytics",
    "Model Performance",
    "Prediction Lab",
    "SARIMA Forecast"
])

# =====================================================
# TAB 1: OVERVIEW
# =====================================================

with tabs[0]:

    st.header("CINEMETRIX Overview")

    if financial is not None:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Movies Analysed",
            f"{len(financial):,}"
        )

        if "budget_adjusted" in financial:
            col2.metric(
                "Average Budget",
                f"${financial['budget_adjusted'].mean():,.0f}"
            )

        if "revenue_adjusted" in financial:
            col3.metric(
                "Average Revenue",
                f"${financial['revenue_adjusted'].mean():,.0f}"
            )

        if "adjusted_profit" in financial:
            col4.metric(
                "Average Adjusted Profit",
                f"${financial['adjusted_profit'].mean():,.0f}"
            )

        if "adjusted_performance" in financial:

            st.subheader("Commercial Performance")

            counts = financial[
                "adjusted_performance"
            ].value_counts()

            st.bar_chart(counts)

    st.divider()

    st.subheader("Research Objective")

    st.write(
        "How can data analytics and artificial intelligence "
        "help us understand audience preferences, analyse "
        "movie performance, and predict commercial success?"
    )

    st.caption(
        "Profit figures represent revenue minus production "
        "budget only, not actual net profit."
    )


# =====================================================
# TAB 2: AUDIENCE ANALYTICS
# =====================================================

with tabs[1]:

    st.header("Audience Intelligence")

    if genre_data is not None:

        st.subheader("Genre Analysis")

        st.dataframe(
            genre_data,
            use_container_width=True
        )

        genre_col = next(
            (c for c in ["genre", "genres"]
             if c in genre_data.columns),
            None
        )

        rating_col = next(
            (c for c in [
                "weighted_average_rating",
                "mean_average_rating",
                "average_rating"
            ] if c in genre_data.columns),
            None
        )

        if genre_col and rating_col:

            chart = genre_data[
                [genre_col, rating_col]
            ].dropna().set_index(genre_col)

            st.bar_chart(chart)

    if yearly_ratings is not None:

        st.subheader("Rating Activity Over Time")

        st.dataframe(
            yearly_ratings,
            use_container_width=True
        )

        year_col = next(
            (c for c in yearly_ratings.columns
             if "year" in c.lower()),
            None
        )

        numeric_cols = yearly_ratings.select_dtypes(
            include=np.number
        ).columns.tolist()

        if year_col and numeric_cols:

            st.line_chart(
                yearly_ratings.set_index(year_col)[numeric_cols]
            )

    st.info(
        "MovieLens records user ratings, not verified movie views. "
        "Rating activity is used as an audience-engagement proxy."
    )


# =====================================================
# TAB 3: FINANCIAL ANALYTICS
# =====================================================

with tabs[2]:

    st.header("Financial Intelligence")

    if financial is not None:

        if "release_year" in financial.columns:

            years = sorted(
                financial["release_year"].dropna().unique()
            )

            selected_years = st.slider(
                "Release year range",
                int(min(years)),
                int(max(years)),
                (int(min(years)), int(max(years)))
            )

            financial_view = financial[
                financial["release_year"].between(
                    selected_years[0],
                    selected_years[1]
                )
            ]

        else:
            financial_view = financial

        if "adjusted_performance" in financial_view:

            st.subheader("Performance Categories")

            st.bar_chart(
                financial_view[
                    "adjusted_performance"
                ].value_counts()
            )

        if {
            "budget_adjusted",
            "revenue_adjusted"
        }.issubset(financial_view.columns):

            st.subheader("Budget vs Revenue")

            st.scatter_chart(
                financial_view,
                x="budget_adjusted",
                y="revenue_adjusted"
            )

        if "adjusted_profit" in financial_view:

            st.subheader("Top 10 Movies by Adjusted Profit")

            columns = [
                c for c in [
                    "title",
                    "release_year",
                    "budget_adjusted",
                    "revenue_adjusted",
                    "adjusted_profit"
                ]
                if c in financial_view.columns
            ]

            st.dataframe(
                financial_view.nlargest(
                    10, "adjusted_profit"
                )[columns],
                use_container_width=True
            )

    if scenario_data is not None:

        st.subheader("Saved Financial Scenarios")

        st.dataframe(
            scenario_data,
            use_container_width=True
        )


# =====================================================
# TAB 4: MODEL PERFORMANCE
# =====================================================

with tabs[3]:

    st.header("Predictive Model Comparison")

    if model_results is not None:

        st.dataframe(
            model_results,
            use_container_width=True
        )

        r2_col = next(
            (c for c in model_results.columns
             if c.lower() in ["r2", "r²", "r2_score"]),
            None
        )

        model_col = next(
            (c for c in model_results.columns
             if "model" in c.lower()),
            None
        )

        if r2_col and model_col:

            st.subheader("R² Comparison")

            st.bar_chart(
                model_results.set_index(model_col)[r2_col]
            )

    st.subheader("Available Trained Models")

    for name in models:
        st.write(f"✓ {name}")

    st.info(
        "Post-release predictions use audience ratings, votes "
        "and popularity. They must not be interpreted as "
        "pre-release predictions."
    )


# =====================================================
# TAB 5: PREDICTION LAB
# =====================================================

with tabs[4]:

    st.header("Interactive Movie Prediction Lab")

    st.write(
        "Enter a hypothetical film profile to generate "
        "predictions using your trained models."
    )

    with st.form("prediction_form"):

        c1, c2, c3 = st.columns(3)

        budget = c1.number_input(
            "Production Budget (USD)",
            min_value=1000.0,
            value=50_000_000.0,
            step=1_000_000.0
        )

        runtime = c2.number_input(
            "Runtime (minutes)",
            min_value=1,
            max_value=400,
            value=120
        )

        release_year = c3.number_input(
            "Release Year",
            min_value=1915,
            max_value=2100,
            value=2026
        )

        release_month = st.selectbox(
            "Release Month",
            list(range(1, 13)),
            index=6
        )

        selected_genres = st.multiselect(
            "Movie Genres",
            GENRES,
            default=["Action", "Science Fiction"]
        )

        st.subheader("Post-release Audience Signals")

        a1, a2, a3 = st.columns(3)

        vote_average = a1.number_input(
            "TMDB Vote Average",
            0.0, 10.0, 7.0
        )

        vote_count = a2.number_input(
            "TMDB Vote Count",
            min_value=0,
            value=1000
        )

        popularity = a3.number_input(
            "TMDB Popularity",
            min_value=0.0,
            value=50.0
        )

        a4, a5 = st.columns(2)

        average_rating = a4.number_input(
            "MovieLens Average Rating",
            0.0, 5.0, 3.8
        )

        rating_count = a5.number_input(
            "MovieLens Rating Count",
            min_value=0,
            value=500
        )

        submit = st.form_submit_button(
            "Run Predictions",
            type="primary"
        )

    if submit:

        inputs = {
            "budget": budget,
            "runtime": runtime,
            "release_year": release_year,
            "release_month": release_month,
            "genres": selected_genres,
            "vote_average": vote_average,
            "vote_count": vote_count,
            "popularity": popularity,
            "average_rating": average_rating,
            "rating_count": rating_count
        }

        base_features = ["budget", "runtime"]

        genre_features = base_features + GENRES

        final_features = (
            base_features
            + ["release_year", "release_month"]
            + GENRES
        )

        post_features = [
            "budget",
            "runtime",
            "vote_average",
            "vote_count",
            "popularity",
            "average_rating",
            "rating_count"
        ]

        known_features = {
            "Ridge Revenue": base_features,
            "Random Forest Revenue": base_features,
            "LightGBM Basic": base_features,
            "LightGBM Genre": genre_features,
            "LightGBM Final": final_features,
            "LightGBM Post-release": post_features,
            "LightGBM Profit": base_features,
            "LightGBM Hit Classifier": base_features
        }

        results = []
        skipped = []

        for name, model in models.items():

            features = getattr(
                model, "feature_names_in_", None
            )

            if features is None and hasattr(model, "booster_"):

                try:
                    features = model.booster_.feature_name()
                except Exception:
                    features = None

            if features is None:
                features = known_features.get(name)

            if features is None:
                skipped.append(name)
                continue

            features = list(features)

            row = {}

            for feature in features:

                if feature == "budget":
                    row[feature] = np.log1p(budget)

                elif feature == "runtime":
                    row[feature] = runtime

                elif feature == "release_year":
                    row[feature] = release_year

                elif feature == "release_month":
                    row[feature] = release_month

                elif feature in GENRES:
                    row[feature] = int(
                        feature in selected_genres
                    )

                elif feature == "vote_average":
                    row[feature] = vote_average

                elif feature == "vote_count":
                    row[feature] = vote_count

                elif feature == "popularity":
                    row[feature] = popularity

                elif feature == "average_rating":
                    row[feature] = average_rating

                elif feature == "rating_count":
                    row[feature] = rating_count

                else:
                    row = None
                    break

            if row is None:
                skipped.append(name)
                continue

            X = pd.DataFrame([row], columns=features)

            # Use DataFrame for models with stored feature names.
            # Otherwise use the known column order.
            has_names = (
                getattr(model, "feature_names_in_", None)
                is not None
            )

            model_input = X if has_names else X.to_numpy()

            try:

                if name == "LightGBM Hit Classifier":

                    prediction = int(
                        model.predict(model_input)[0]
                    )

                    probability = None

                    if hasattr(model, "predict_proba"):

                        probabilities = model.predict_proba(
                            model_input
                        )[0]

                        classes = list(
                            getattr(model, "classes_", [0, 1])
                        )

                        if 1 in classes:
                            probability = probabilities[
                                classes.index(1)
                            ]

                    results.append({
                        "Model": name,
                        "Prediction": (
                            f"Hit probability: {probability:.1%}"
                            if probability is not None
                            else f"Class: {prediction}"
                        )
                    })

                else:

                    prediction = float(
                        np.asarray(
                            model.predict(model_input)
                        ).reshape(-1)[0]
                    )

                    if name in [
                        "Ridge Revenue",
                        "Random Forest Revenue",
                        "LightGBM Basic",
                        "LightGBM Genre",
                        "LightGBM Final",
                        "LightGBM Post-release"
                    ]:

                        prediction = np.expm1(prediction)

                        label = (
                            f"Predicted Revenue: "
                            f"${prediction:,.0f}"
                        )

                    elif name == "LightGBM Profit":

                        label = (
                            f"Predicted Profit Before Other Costs: "
                            f"${prediction:,.0f}"
                        )

                    else:
                        label = f"{prediction:,.2f}"

                    results.append({
                        "Model": name,
                        "Prediction": label
                    })

            except Exception as error:
                skipped.append(f"{name}: {error}")

        if results:

            st.subheader("Prediction Results")

            st.dataframe(
                pd.DataFrame(results),
                use_container_width=True
            )

        if skipped:

            with st.expander("Models not evaluated"):

                for item in skipped:
                    st.write(item)

        st.warning(
            "These are estimates, not guaranteed outcomes. "
            "The hit classifier uses the project's revenue-to-budget "
            "threshold of 2. The profit model excludes other costs."
        )


# =====================================================
# TAB 6: SARIMA FORECAST
# =====================================================

with tabs[5]:

    st.header("Audience Activity Forecasting")

    st.caption(
        "Forecasts monthly MovieLens rating activity, "
        "not actual movie viewing figures."
    )

    monthly_path = find_file(
        PROJECT,
        "monthly_ratings.csv"
    )

    if monthly_path is not None:

        monthly = pd.read_csv(monthly_path)

        date_col = next(
            (c for c in monthly.columns
             if c.lower() in ["date", "month"]),
            monthly.columns[0]
        )

        value_col = next(
            (c for c in monthly.columns
             if c.lower() in ["rating_count", "count"]),
            None
        )

        if value_col is None:

            numeric = monthly.select_dtypes(
                include=np.number
            ).columns.tolist()

            value_col = numeric[0] if numeric else None

        if value_col is not None:

            monthly[date_col] = pd.to_datetime(
                monthly[date_col]
            )

            series = (
                monthly.set_index(date_col)[value_col]
                .astype(float)
                .sort_index()
                .asfreq("MS")
                .fillna(0)
            )

            st.subheader("Historical Rating Activity")

            st.line_chart(series)

            horizon = st.slider(
                "Forecast Horizon (Months)",
                min_value=3,
                max_value=24,
                value=12
            )

            if st.button(
                "Generate SARIMA Forecast",
                type="primary"
            ):

                with st.spinner("Generating forecast..."):

                    model = SARIMAX(
                        series,
                        order=(1, 0, 1),
                        seasonal_order=(1, 0, 1, 12),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )

                    fitted = model.fit(disp=False)

                    forecast = fitted.get_forecast(
                        steps=horizon
                    )

                    forecast_df = forecast.summary_frame()

                    st.subheader("Forecast Results")

                    chart = pd.DataFrame({
                        "Historical": series.tail(36),
                        "Forecast": forecast_df["mean"]
                    })

                    st.line_chart(chart)

                    st.dataframe(
                        forecast_df.round(0),
                        use_container_width=True
                    )

    else:

        st.warning(
            "monthly_ratings.csv was not found in the ZIP. "
            "Save it from Colab and rebuild your backup ZIP."
        )

        if sarima_saved is not None:

            st.subheader("Saved SARIMA Test Forecasts")

            st.dataframe(
                sarima_saved,
                use_container_width=True
            )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "CINEMETRIX | Research and Educational Analytics Project"
)
