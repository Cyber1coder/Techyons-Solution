import json
import re
import os
import uuid
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, static_folder=".")
CORS(app)  # Enable Cross-Origin Resource Sharing

DATA_PATH = "compensation_dataset_1250_entries.json"
NEW_ENTRY_PATH = "new_entry.json"

# Configurations derived from Colab notebook
MIN_SEGMENT_SIZE = 3
LOF_N_NEIGHBORS = 5
LOF_CONTAMINATION = 0.10
CONSISTENCY_THRESHOLD = 0.50

WEIGHT_RULE = 0.40
WEIGHT_LOF = 0.60

PUBLIC_COMPANIES = {
    "oracle", "microsoft", "google", "amazon", "apple", "meta", "netflix",
    "salesforce", "servicenow", "snowflake", "adobe", "ibm", "intel",
    "deere & company", "cigna", "hca healthcare", "jpmorgan", "goldman sachs",
    "morgan stanley", "bank of america", "wells fargo", "visa", "mastercard",
    "unitedhealth", "cvs", "walmart", "target", "boeing", "lockheed martin",
    "accenture", "infosys", "wipro", "tcs", "capgemini",
}
STARTUP_STAGES = {"seed", "series a", "series b", "series c", "series d"}

LEVEL_ORDINAL = {
    "l1": 1, "l2": 2, "l3": 3, "l4": 4, "l5": 5, "l6": 6, "l7": 7,
    "ic1": 1, "ic2": 2, "ic3": 3, "ic4": 4, "ic5": 5, "ic6": 6,
    "e1": 1, "e2": 2, "e3": 3, "e4": 4, "e5": 5, "e6": 6, "e7": 7,
    "junior": 1, "mid": 2, "mid-level": 2, "senior": 3,
    "staff": 4, "principal": 5, "fellow": 6, "distinguished": 7,
    "swe i": 1, "swe ii": 2, "senior swe": 3, "staff swe": 4,
    "m1": 1, "m2": 2, "m3": 3, "m4": 4, "m5": 5,
    "manager": 1, "senior manager": 2, "director": 3,
    "senior director": 4, "vp": 5, "svp": 6, "evp": 7, "cto": 8,
    "partner": 3,
}

FAANG = {"google", "meta", "amazon", "apple", "microsoft", "netflix"}
TIER2 = {
    "oracle", "salesforce", "servicenow", "snowflake", "adobe", "nvidia",
    "uber", "lyft", "airbnb", "stripe", "linkedin", "twitter", "x",
    "databricks", "confluent", "figma", "notion", "openai", "anthropic",
}

RULE_COLUMNS = [
    "flag_yal_gt_yac",
    "flag_yac_gt_yoe",
    "flag_negative_experience",
    "flag_zero_or_negative_base",
    "flag_tc_less_than_base",
    "flag_junior_level_high_yoe",
    "flag_bonus_pct_mismatch",
    "flag_public_co_startup_stage",
    "flag_tc_arithmetic_mismatch",
    "flag_companysize_mismatch",
]

BENCHMARKS = {
    ("Software Engineer", "junior", "US"): {"min": 90000, "max": 220000},
    ("Software Engineer", "senior", "US"): {"min": 180000, "max": 500000}
}

COUNTRY_REGION = {
    1: "US", 113: "India", 826: "UK", 276: "Germany",
    250: "France", 380: "Italy", 724: "Spain", 124: "Canada",
    36: "Australia", 392: "Japan", 156: "China", 76: "Brazil"
}

# In-Memory Database variables
master_dataset_list = []

# --- Robust helper mappings ---
def get_exchange_rate(currency):
    rates = {
        "USD": 1.0,
        "INR": 83.94,
        "EUR": 0.92,
        "GBP": 0.78,
        "CAD": 1.36,
        "AUD": 1.50
    }
    return rates.get(str(currency).upper().strip(), 1.0)

def map_country_id_from_location(location):
    if not isinstance(location, str):
        return -1
    loc_lower = location.lower().strip()
    if "usa" in loc_lower or "united states" in loc_lower or ", ca" in loc_lower or ", ny" in loc_lower or ", tx" in loc_lower or ", wa" in loc_lower:
        return 1
    if "india" in loc_lower or "bangalore" in loc_lower or "hyderabad" in loc_lower or "mumbai" in loc_lower or "delhi" in loc_lower or "pune" in loc_lower:
        return 113
    if "united kingdom" in loc_lower or " uk" in loc_lower or "london" in loc_lower or "england" in loc_lower:
        return 826
    if "germany" in loc_lower or "berlin" in loc_lower or "munich" in loc_lower or "frankfurt" in loc_lower:
        return 276
    if "france" in loc_lower or "paris" in loc_lower:
        return 250
    if "italy" in loc_lower or "rome" in loc_lower or "milan" in loc_lower:
        return 380
    if "spain" in loc_lower or "madrid" in loc_lower or "barcelona" in loc_lower:
        return 724
    if "canada" in loc_lower or "toronto" in loc_lower or "vancouver" in loc_lower or "montreal" in loc_lower:
        return 124
    if "australia" in loc_lower or "sydney" in loc_lower or "melbourne" in loc_lower:
        return 36
    if "japan" in loc_lower or "tokyo" in loc_lower or "osaka" in loc_lower:
        return 392
    if "china" in loc_lower or "beijing" in loc_lower or "shanghai" in loc_lower:
        return 156
    if "brazil" in loc_lower or "sao paulo" in loc_lower or "rio de janeiro" in loc_lower:
        return 76
    return -1

def slugify(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    return text

def apply_robust_mappings(df):
    """
    Applies exchangeRate, countryId, and annualTargetBonusValue mappings
    so standard algorithms run flawlessly on compensation_dataset_1250_entries.json
    """
    df = df.copy()
    
    # Map exchange rate
    if "exchangeRate" not in df.columns:
        df["exchangeRate"] = 1.0
    
    curr_col = df.get("baseSalaryCurrency", pd.Series("USD", index=df.index))
    mapped_exch = curr_col.apply(get_exchange_rate)
    df["exchangeRate"] = df["exchangeRate"].fillna(mapped_exch)
    
    # Map countryId based on location string
    if "countryId" not in df.columns:
        df["countryId"] = -1
    
    loc_col = df.get("location", pd.Series("", index=df.index))
    mapped_cid = loc_col.apply(map_country_id_from_location)
    df["countryId"] = df["countryId"].replace(-1, np.nan).fillna(mapped_cid).astype(int)

    # Map annualTargetBonusValue to avgAnnualBonusValue if missing
    if "annualTargetBonusValue" not in df.columns:
        df["annualTargetBonusValue"] = df.get("avgAnnualBonusValue", pd.Series(0, index=df.index))
    df["annualTargetBonusValue"] = df["annualTargetBonusValue"].fillna(df.get("avgAnnualBonusValue", 0))

    return df

# --- Pipeline functions extracted from Colab script ---

def flag_rules(df: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame(index=df.index)

    yoe = pd.to_numeric(df.get("yearsOfExperience", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    yac = pd.to_numeric(df.get("yearsAtCompany",    pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    yal = pd.to_numeric(df.get("yearsAtLevel",      pd.Series(0, index=df.index)), errors="coerce").fillna(0)

    flags["flag_yal_gt_yac"]          = yal > yac
    flags["flag_yac_gt_yoe"]          = yac > yoe
    flags["flag_negative_experience"] = (yoe < 0) | (yac < 0) | (yal < 0)

    flags["flag_zero_or_negative_base"] = pd.to_numeric(
        df.get("baseSalary", pd.Series(0, index=df.index)), errors="coerce"
    ).fillna(0) <= 0

    flags["flag_tc_less_than_base"]       = False
    flags["flag_tc_arithmetic_mismatch"]  = False

    level_norm = df.get("level", pd.Series("", index=df.index, dtype=str)).str.lower().str.strip().fillna("")
    is_entry   = level_norm.isin(["l1", "ic1", "e1", "junior", "swe i"])
    flags["flag_junior_level_high_yoe"] = is_entry & (yoe > 8)

    base_raw     = pd.to_numeric(df.get("baseSalary", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    bonus_pct    = pd.to_numeric(df.get("annualTargetBonusPercentage", pd.Series(0, index=df.index)), errors="coerce").fillna(0) / 100
    bonus_actual = pd.to_numeric(df.get("annualTargetBonusValue", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    expected_bonus = base_raw * bonus_pct
    both_nonzero   = (expected_bonus > 0) & (bonus_actual > 0)
    ratio = (bonus_actual / expected_bonus.replace(0, np.nan)).fillna(1.0)
    flags["flag_bonus_pct_mismatch"] = both_nonzero & ((ratio < 0.75) | (ratio > 1.25))

    company_lc = df.get("company", pd.Series("", index=df.index, dtype=str)).str.lower().str.strip().fillna("")
    funding_lc = df.get("fundingStage", pd.Series("", index=df.index, dtype=str)).str.lower().str.strip().fillna("")
    flags["flag_public_co_startup_stage"] = (
        company_lc.isin(PUBLIC_COMPANIES) & funding_lc.isin(STARTUP_STAGES)
    )

    size_map = {
        "1-10": 5, "11-50": 30, "51-200": 125, "201-500": 350,
        "501-1000": 750, "1001-5000": 3000, "5000+": 10000,
    }
    size_numeric = df.get("companySize", pd.Series(np.nan, index=df.index, dtype=str)).map(size_map).fillna(np.nan)
    flags["flag_companysize_mismatch"] = (
        company_lc.isin(PUBLIC_COMPANIES) & (size_numeric < 1000)
    )

    return flags

def normalize_salaries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    exch = pd.to_numeric(df.get("exchangeRate", pd.Series(1.0, index=df.index)), errors="coerce").replace(0, np.nan).fillna(1.0)
    base_currency  = df.get("baseSalaryCurrency", pd.Series("USD", index=df.index, dtype=str)).fillna("USD")
    bonus_currency = df.get("bonusCurrency",      pd.Series("USD", index=df.index, dtype=str)).fillna("USD")

    base_raw  = pd.to_numeric(df.get("baseSalary",            pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    bonus_raw = pd.to_numeric(df.get("avgAnnualBonusValue",   pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)

    df["baseSalary_USD"] = np.where(base_currency == "USD", base_raw, base_raw / exch)
    df["totalCompensation_USD"] = pd.to_numeric(
        df.get("totalCompensation", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)

    # If currency of TC is non-USD, normalize it too (since totalCompensation might be local in some entries)
    # Wait, in the Colab dataset totalCompensation is USD already, but let's check baseSalaryCurrency to be safe
    # If the user currency is EUR or INR, totalCompensation is likely in local currency too. Let's normalise if they match.
    df["totalCompensation_USD"] = np.where(base_currency == "USD", df["totalCompensation_USD"], df["totalCompensation_USD"] / exch)

    df["avgAnnualBonusValue_USD"] = np.where(
        bonus_currency == "USD", bonus_raw, bonus_raw / exch
    )

    df["avgAnnualStockGrantValue_USD"] = pd.to_numeric(
        df.get("avgAnnualStockGrantValue", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)

    return df

def fill_postnorm_flags(df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    flags["flag_tc_less_than_base"] = (
        (df["totalCompensation_USD"] < df["baseSalary_USD"]) &
        (df["baseSalary_USD"] > 0)
    )

    reconstructed = (
        df["baseSalary_USD"] +
        df["avgAnnualBonusValue_USD"] +
        df["avgAnnualStockGrantValue_USD"]
    )
    tc_safe   = df["totalCompensation_USD"].replace(0, np.nan)
    deviation = ((reconstructed - tc_safe).abs() / tc_safe).fillna(0)
    flags["flag_tc_arithmetic_mismatch"] = deviation > 0.15

    return flags

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    level_key  = df.get("level", pd.Series("", index=df.index, dtype=str)).str.lower().str.strip().fillna("")
    df["level_ordinal"] = level_key.map(LEVEL_ORDINAL).fillna(0).astype(int)

    def _level_group(o):
        if o <= 1: return "junior"
        if o <= 2: return "mid"
        if o <= 3: return "senior"
        return "staff_plus"
    df["level_group"] = df["level_ordinal"].apply(_level_group)

    cid = pd.to_numeric(df.get("countryId", pd.Series(-1, index=df.index)), errors="coerce").fillna(-1).astype(int)
    df["region"] = cid.map(COUNTRY_REGION).fillna("Other")

    df["jobFamily_clean"] = (
        df.get("jobFamily", pd.Series("Unknown", index=df.index, dtype=str))
          .str.strip().str.title().fillna("Unknown")
    )

    def _tier(name):
        n = str(name).lower().strip()
        if n in FAANG:  return "faang"
        if n in TIER2:  return "tier2"
        return "other"
    df["company_tier"] = df.get("company", pd.Series("other", index=df.index, dtype=str)).apply(_tier)

    emp = df.get("employmentType", pd.Series("unknown", index=df.index, dtype=str)).str.lower().str.strip().fillna("unknown")
    df["employmentType_clean"] = emp

    base_safe = df["baseSalary_USD"].replace(0, np.nan)
    tc_safe   = df["totalCompensation_USD"].replace(0, np.nan)

    df["bonus_ratio"] = (df["avgAnnualBonusValue_USD"]      / base_safe).fillna(0).clip(0, 5)
    df["stock_ratio"] = (df["avgAnnualStockGrantValue_USD"] / base_safe).fillna(0).clip(0, 20)
    df["tc_to_base"]  = (tc_safe / base_safe).fillna(1).clip(0.5, 20)

    df["log_base_salary"] = np.log1p(df["baseSalary_USD"])
    df["log_tc"]          = np.log1p(df["totalCompensation_USD"])

    return df

LOF_FEATURES = [
    "log_base_salary",
    "log_tc",
    "avgAnnualBonusValue_USD",
    "avgAnnualStockGrantValue_USD",
    "yearsOfExperience",
    "bonus_ratio",
    "stock_ratio",
    "tc_to_base",
]

def run_lof(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["segment"] = (
        df["jobFamily_clean"] + " | " +
        df["level_group"]     + " | " +
        df["region"]
    )
    df["segment_size"] = 0
    df["lof_score"]    = np.nan
    df["lof_label"]    = "unscored"

    for seg_label, group in df.groupby("segment"):
        idx = group.index
        df.loc[idx, "segment_size"] = len(group)

        if len(group) < MIN_SEGMENT_SIZE:
            continue

        X = group[LOF_FEATURES].fillna(0).values
        X_scaled = StandardScaler().fit_transform(X)

        k = min(LOF_N_NEIGHBORS, len(group) - 1)
        lof = LocalOutlierFactor(
            n_neighbors=k,
            contamination=LOF_CONTAMINATION,
            novelty=False,
        )
        labels = lof.fit_predict(X_scaled)
        scores = -lof.negative_outlier_factor_

        df.loc[idx, "lof_score"] = scores
        df.loc[idx, "lof_label"] = np.where(labels == -1, "outlier", "inlier")

    return df

def compute_scores(df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in RULE_COLUMNS:
        df[col] = flags[col].astype(bool).values

    df["rule_violations"] = flags[RULE_COLUMNS].astype(int).sum(axis=1).values
    df["rule_score"]      = 1.0 - (df["rule_violations"] / len(RULE_COLUMNS))

    lof_raw = df["lof_score"].copy()
    df["lof_normalized"] = 1.0 - ((lof_raw.clip(1.0, 3.0) - 1.0) / 2.0)

    has_lof = df["lof_score"].notna()

    df["consistency_score"] = np.where(
        has_lof,
        WEIGHT_RULE * df["rule_score"] + WEIGHT_LOF * df["lof_normalized"],
        df["rule_score"],
    ).clip(0.0, 1.0).round(4)

    return df

def detect_duplicates(df):
    df = df.copy()
    cols = ["company", "title", "level", "jobFamily", "baseSalary", "yearsOfExperience", "location"]
    
    combined = (
        df[cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(combined)
    similarity = cosine_similarity(matrix)

    flags = []
    for i in range(len(df)):
        duplicate = False
        for j in range(len(df)):
            if i != j and similarity[i][j] > 0.95:
                duplicate = True
                break
        flags.append(duplicate)

    df["flag_duplicate"] = flags
    return df

def benchmark_validation(df):
    flags = []
    for _, row in df.iterrows():
        key = (
            row["jobFamily_clean"],
            row["level_group"],
            row["region"]
        )
        salary = row["baseSalary_USD"]

        if key in BENCHMARKS:
            low = BENCHMARKS[key]["min"]
            high = BENCHMARKS[key]["max"]
            flags.append(salary < low or salary > high)
        else:
            flags.append(False)

    df["flag_benchmark"] = flags
    return df

def user_trust_score(df):
    df = df.copy()
    trust = []
    for _, row in df.iterrows():
        score = row["consistency_score"]
        duplicate = row.get("flag_duplicate", False)
        benchmark = row.get("flag_benchmark", False)

        trust_score = 1.0
        if duplicate:
            trust_score -= 0.3
        if benchmark:
            trust_score -= 0.3
        if score < 0.7:
            trust_score -= 0.2

        trust.append(max(0, round(trust_score, 2)))

    df["user_trust"] = trust
    return df

def final_decision(df):
    decisions = []
    for _, row in df.iterrows():
        score = row["consistency_score"]
        trust = row["user_trust"]
        duplicate = row["flag_duplicate"]
        benchmark = row["flag_benchmark"]

        if duplicate:
            decisions.append("Review")
        elif benchmark:
            decisions.append("Review")
        elif score > 0.9 and trust > 0.8:
            decisions.append("Approve")
        elif score < 0.5:
            decisions.append("Reject")
        else:
            decisions.append("Review")

    df["decision"] = decisions
    return df

def explain_flags(row):
    reasons = []
    if row["flag_duplicate"]:
        reasons.append("Duplicate salary entry matches other records excessively")
    if row["flag_benchmark"]:
        reasons.append("Salary falls outside the industry baseline standard benchmarks")
    if row["lof_label"] == "outlier":
        reasons.append("LOF segment anomaly: compensation ratios/tenure do not align with peers")
    if row["flag_yal_gt_yac"]:
        reasons.append("Years at level exceeds years at company (impossible)")
    if row["flag_yac_gt_yoe"]:
        reasons.append("Years at company exceeds total years of experience (impossible)")
    if row["flag_negative_experience"]:
        reasons.append("Experience/tenure inputs cannot be negative values")
    if row["flag_zero_or_negative_base"]:
        reasons.append("Base salary cannot be zero or negative")
    if row["flag_tc_less_than_base"]:
        reasons.append("Total compensation is less than base salary")
    if row["flag_junior_level_high_yoe"]:
        reasons.append("Junior/Entry level lists suspiciously high years of experience (> 8 years)")
    if row["flag_bonus_pct_mismatch"]:
        reasons.append("Declared bonus percentage deviates significantly (>25%) from bonus values")
    if row["flag_public_co_startup_stage"]:
        reasons.append("Public company listed with startup funding stage tag")
    if row["flag_tc_arithmetic_mismatch"]:
        reasons.append("Total compensation math doesn't sum up with base, bonus, and stock grants (>15% error)")
    if row["flag_companysize_mismatch"]:
        reasons.append("Large public company is listed with a tiny company size headcount (<200 employees)")

    return "; ".join(reasons)

# --- Startup: Load Master Dataset ---
def load_master_dataset():
    global master_dataset_list
    if os.path.exists(DATA_PATH):
        try:
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                master_dataset_list = json.load(f)
            print(f"[load] Loaded {len(master_dataset_list)} master records from {DATA_PATH}")
        except Exception as e:
            print(f"[error] Failed to load {DATA_PATH}: {e}")
            master_dataset_list = []
    else:
        print(f"[warning] Dataset file {DATA_PATH} not found!")
        master_dataset_list = []

load_master_dataset()

# --- Static routing endpoints ---
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(".", path)

# --- API Endpoint: Classify Submission ---
@app.route("/classify", methods=["POST"])
def classify():
    try:
        form_data = request.json
        if not form_data:
            return jsonify({"error": "Missing input body"}), 400

        # 1. Map to exact schema required by dataset
        company = form_data.get("company", "").strip()
        title = form_data.get("title", "").strip()
        jobFamily = form_data.get("jobFamily", "Software Engineer")
        level = form_data.get("level", "L1").strip()
        
        company_slug = slugify(company)
        jobFamilySlug = slugify(jobFamily)

        base_salary = float(form_data.get("baseSalary", 0))
        currency = form_data.get("baseSalaryCurrency", "USD")
        stock_grant = float(form_data.get("avgAnnualStockGrantValue", 0))
        bonus_value = float(form_data.get("avgAnnualBonusValue", 0))
        target_bonus_pct = float(form_data.get("annualTargetBonusPercentage", 0))
        total_comp = base_salary + stock_grant + bonus_value

        new_entry = {
            "uuid": str(uuid.uuid4()),
            "company": company,
            "title": title,
            "jobFamily": jobFamily,
            "jobFamilySlug": jobFamilySlug,
            "level": level,
            "focusTag": form_data.get("focusTag") or jobFamily,
            "yearsOfExperience": int(form_data.get("yearsOfExperience", 0)),
            "yearsAtCompany": int(form_data.get("yearsAtCompany", 0)),
            "yearsAtLevel": int(form_data.get("yearsAtLevel", 0)),
            "offerDate": form_data.get("offerDate", ""),
            "location": form_data.get("location", ""),
            "workArrangement": form_data.get("workArrangement", "onsite"),
            "compPerspective": form_data.get("compPerspective", "offer"),
            "baseSalary": base_salary,
            "baseSalaryCurrency": currency,
            "employmentType": form_data.get("employmentType", "full-time"),
            "totalCompensation": total_comp,
            "avgAnnualStockGrantValue": stock_grant,
            "avgAnnualBonusValue": bonus_value,
            "gender": form_data.get("gender") or None,
            "ethnicity": None,
            "education": form_data.get("education", "Bachelor's"),
            "companyInfo": {
                "registered": True,
                "name": company,
                "slug": company_slug
            },
            "annualTargetBonusPercentage": target_bonus_pct,
            "userCurrency": currency
        }

        # 2. Save results in the JSON format into new_entry.json as requested
        with open(NEW_ENTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(new_entry, f, indent=2)
        print(f"[saved] Single record written to {NEW_ENTRY_PATH}")

        # 3. Create DataFrame from master dataset list + new entry
        combined_list = master_dataset_list + [new_entry]
        df = pd.json_normalize(combined_list)

        # 4. Apply our dynamic, robust maps (exchange rates, region mapping, bonus mapping)
        df = apply_robust_mappings(df)

        # 5. Run standard classification pipeline
        flags = flag_rules(df)
        df = normalize_salaries(df)
        flags = fill_postnorm_flags(df, flags)
        df = engineer_features(df)
        df = detect_duplicates(df)
        df = benchmark_validation(df)
        df = run_lof(df)
        df = compute_scores(df, flags)
        df = user_trust_score(df)
        df = final_decision(df)
        
        # Apply flag explanations
        df["explanation"] = df.apply(explain_flags, axis=1)

        # 6. Extract classification variables of our newly added row (the last row)
        result_row = df.iloc[-1]

        # Calculate segment context for visual comparison
        segment = result_row["segment"]
        segment_df = df[df["segment"] == segment]
        segment_size = int(result_row["segment_size"])
        
        segment_avg_salary = None
        if segment_size >= 1:
            segment_avg_salary = float(segment_df["baseSalary_USD"].mean())

        # Generate response payload
        res_payload = {
            "decision": str(result_row["decision"]),
            "consistency_score": float(result_row["consistency_score"]),
            "user_trust": float(result_row["user_trust"]),
            "lof_label": str(result_row["lof_label"]),
            "lof_score": float(result_row["lof_score"]) if pd.notna(result_row["lof_score"]) else None,
            "explanation": str(result_row["explanation"]),
            "segment": segment,
            "segment_size": segment_size,
            "segment_avg_salary": segment_avg_salary
        }

        return jsonify(res_payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
