#!/usr/bin/env python3

"""
Enhanced Interactive CLI Dashboard for Retail Sales Forecasting
includes:
 - Rich-powered navigation menu
 - Numeric option selection with validation
 - Auto-return to Home Menu after every action
"""

import sys
from pathlib import Path
import json
import pandas as pd
from joblib import load
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt

# Typer app (but we also override entry with our own menu)
app = typer.Typer(add_completion=False)
console = Console()

# ----------------------------------------------------------
# PATH + CONFIG
# ----------------------------------------------------------

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))   # allow "from config import ..." from project root

from config import (
    METRICS_DIR,
    FEATURE_DATA_PATH,
    MERGED_DATA_PATH,
    MODELS_DIR,
    DATE_COLUMN,
    STORE_COLUMN,
    FAMILY_COLUMN,
    TARGET_COLUMN,
)

# ----------------------------------------------------------
# BANNER
# ----------------------------------------------------------

def banner():
    console.print(
        Panel.fit(
            "[bold cyan]Retail Sales Forecasting UI[/]\n"
            "[green]Kaggle Store Sales + NOAA Weather[/]",
            border_style="cyan"
        )
    )

# ----------------------------------------------------------
# LOADER FUNCTIONS
# ----------------------------------------------------------

def load_metrics():
    path = METRICS_DIR / "all_metrics.json"
    return json.load(open(path)) if path.exists() else {}

def load_val_predictions():
    path = METRICS_DIR / "val_predictions.csv"
    return pd.read_csv(path, parse_dates=[DATE_COLUMN]) if path.exists() else pd.DataFrame()

def load_features():
    return pd.read_parquet(FEATURE_DATA_PATH) if FEATURE_DATA_PATH.exists() else pd.DataFrame()

def load_merged():
    return pd.read_parquet(MERGED_DATA_PATH) if MERGED_DATA_PATH.exists() else pd.DataFrame()

def load_models():
    models = {}
    if MODELS_DIR.exists():
        for p in MODELS_DIR.glob("*.joblib"):
            try:
                models[p.stem] = load(p)
            except Exception:
                pass
    return models

# ----------------------------------------------------------
# SMALL HELPER: SAFE VALUE
# ----------------------------------------------------------

def safe_value(df: pd.DataFrame, col: str, default: float) -> float:
    """Return default if column missing OR value is NaN OR df is empty."""
    if df.empty or col not in df.columns:
        return default
    val = df[col].iloc[0]
    if pd.isna(val):
        return default
    return float(val)

# ----------------------------------------------------------
# METRICS VIEW
# ----------------------------------------------------------

def view_metrics():
    banner()
    metrics = load_metrics()
    if not metrics:
        console.print("[red]No metrics found. Run python3 main.py first.[/]")
        return

    table = Table(title="Model Performance Metrics", style="cyan")
    table.add_column("Model", style="bold magenta")
    table.add_column("MAE", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("MAPE (%)", justify="right")

    for model, vals in metrics.items():
        table.add_row(
            model,
            f"{vals['MAE']:.2f}",
            f"{vals['RMSE']:.2f}",
            f"{vals['MAPE']:.2f}",
        )

    console.print(table)

# ----------------------------------------------------------
# BROWSE VALIDATION
# ----------------------------------------------------------

def browse_predictions():
    banner()
    df = load_val_predictions()
    if df.empty:
        console.print("[red]No validation predictions found.[/]")
        return

    console.print("[bold cyan]Sample validation predictions (first 20 rows):[/]")
    console.print(df.head(20))

# ----------------------------------------------------------
# WEATHER VIEW
# ----------------------------------------------------------

def weather_sales():
    banner()
    df = load_merged()
    if df.empty:
        console.print("[red]No merged dataset found.[/]")
        return

    cols = [DATE_COLUMN, STORE_COLUMN, FAMILY_COLUMN, TARGET_COLUMN]
    for c in ["tavg", "prcp"]:
        if c in df.columns:
            cols.append(c)

    console.print("[bold green]Sample weather + sales rows (first 20):[/]")
    console.print(df[cols].head(20))

# ----------------------------------------------------------
# WHAT-IF SIMULATION
# ----------------------------------------------------------

def simulate():
    banner()
    features = load_features()
    models = load_models()

    if features.empty:
        console.print("[red]Features missing — run python3 main.py first.[/]")
        return

    if not models:
        console.print("[red]No trained models found in the models/ folder.[/]")
        return

    # List models
    console.print("\n[bold cyan]Available Models:[/]")
    model_names = list(models.keys())
    for i, m in enumerate(model_names, start=1):
        console.print(f"{i}. {m}")

    # Numeric selection with validation
    while True:
        try:
            selection = int(Prompt.ask("\nChoose model number"))
            if 1 <= selection <= len(model_names):
                model_name = model_names[selection - 1]
                break
            else:
                console.print("[red]Invalid option. Try again.[/]")
        except Exception:
            console.print("[red]Please enter a valid number.[/]")

    model = models[model_name]

    # Store selection
    while True:
        try:
            store = int(Prompt.ask("Enter store number"))
            break
        except Exception:
            console.print("[red]Store must be a number. Try again.[/]")

    # Family selection
    family = Prompt.ask("Enter product family")

    base = (
        features[
            (features[STORE_COLUMN] == store)
            & (features[FAMILY_COLUMN] == family)
        ]
        .sort_values(DATE_COLUMN)
        .tail(1)
    )

    if base.empty:
        console.print("[red]No feature rows for this store/family combination.[/]")
        return

    # Use safe defaults when any weather is missing/NaN
    tavg0 = safe_value(base, "tavg", 20.0)
    prcp0 = safe_value(base, "prcp", 0.0)
    tmax0 = safe_value(base, "tmax", tavg0 + 3.0)
    tmin0 = safe_value(base, "tmin", tavg0 - 3.0)

    console.print(
        f"\n[bold green]Current Conditions (baseline row):[/]\n"
        f"• tavg = {tavg0}\n"
        f"• tmax = {tmax0}\n"
        f"• tmin = {tmin0}\n"
        f"• prcp = {prcp0}\n"
    )

    # New weather – keep prompting until numbers are valid
    while True:
        try:
            tavg_new = float(
                Prompt.ask(
                    "Enter new Avg Temperature (°C)",
                    default=str(round(tavg0, 1)),
                )
            )
            prcp_new = float(
                Prompt.ask(
                    "Enter new Rainfall (mm)",
                    default=str(round(prcp0, 1)),
                )
            )
            break
        except Exception:
            console.print("[red]Invalid numeric input. Try again.[/]")

    # Build simulated feature row
    sim = base.copy()
    sim["tavg"] = tavg_new
    sim["tmax"] = tavg_new + 3.0
    sim["tmin"] = tavg_new - 3.0
    sim["prcp"] = prcp_new
    if "temp_x_prcp" in sim.columns:
        sim["temp_x_prcp"] = sim["tavg"] * sim["prcp"]

    pred = float(model.predict(sim)[0])
    actual = float(base[TARGET_COLUMN].iloc[0])
    diff = pred - actual

    console.print(
        Panel.fit(
            f"[cyan]Baseline: {actual:,.2f}\n"
            f"[magenta]Prediction: {pred:,.2f}\n"
            f"[green]Change: {diff:+,.2f}[/]",
            title=f"Simulation Result ({model_name})",
            border_style="yellow",
        )
    )

# ----------------------------------------------------------
# HOME MENU
# ----------------------------------------------------------

def home_menu():
    """Interactive menu that loops until user exits."""

    while True:
        banner()

        console.print("\n[bold cyan]Choose an option:[/]")
        console.print("1. View Metrics")
        console.print("2. Browse Predictions")
        console.print("3. Weather & Sales")
        console.print("4. What-if Simulator")
        console.print("5. Exit")

        choice = Prompt.ask("\nEnter choice")

        if choice == "1":
            view_metrics()
        elif choice == "2":
            browse_predictions()
        elif choice == "3":
            weather_sales()
        elif choice == "4":
            simulate()
        elif choice == "5":
            console.print("[bold green]Goodbye![/]")
            sys.exit(0)
        else:
            console.print("[red]❌ Invalid option. Try again.[/]")

        # Pause before returning
        console.print("\nPress Enter to return to menu...")
        input("")

# ----------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------

if __name__ == "__main__":
    home_menu()
